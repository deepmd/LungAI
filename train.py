import argparse
import logging
import os
import time

import torch
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Identity,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter

from losses import build_loss
from optimizers import build_optimizer
from models import build_model
from data import LoadImageAndPickled, CropLungWithModeld, get_data_files, check_transform
from utils import *


def build_transforms(cfg):
    train_transforms = Compose(
        [
            LoadImageAndPickled(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            (CropLungWithModeld(keys=["image"]) if cfg.crop_lung else Identity()),
            ScaleIntensityRanged(
                keys=["image"], a_min=cfg.intensity_range[0], a_max=cfg.intensity_range[1],
                b_min=0.0, b_max=1.0, clip=True,
            ),
            # todo: zero-mean if required
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.patch_size,
                pos=cfg.pos_neg_ratio,
                neg=1,
                num_samples=cfg.patch_per_sample,
                image_key="image",
                image_threshold=0,
            ),
            (RandFlipd(keys=["image", "label"],
                       spatial_axis=[0], prob=0.10) if cfg.flip else Identity()),
            (RandFlipd(keys=["image", "label"],
                       spatial_axis=[1], prob=0.10) if cfg.flip else Identity()),
            (RandFlipd(keys=["image", "label"],
                       spatial_axis=[2], prob=0.10) if cfg.flip else Identity()),
            (RandRotate90d(keys=["image", "label"],
                           prob=0.10, max_k=3) if cfg.rotate90 else Identity()),
            (RandShiftIntensityd(keys=["image"],
                                 offsets=0.10, prob=0.50) if cfg.shift_intensity else Identity())

            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=cfg.patch_size,
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImageAndPickled(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            (CropLungWithModeld(keys=["image"]) if cfg.crop_lung else Identity()),
            ScaleIntensityRanged(
                keys=["image"], a_min=cfg.intensity_range[0], a_max=cfg.intensity_range[1],
                b_min=0.0, b_max=1.0, clip=True,
            ),
            # todo: zero-mean if required
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
        ]
    )
    return train_transforms, val_transforms


def build_loaders(cfg):
    train_transforms, val_transforms = build_transforms(cfg)
    train_files = get_data_files(cfg.image_dir, cfg.label_dir, cfg.train_ids)
    val_files = get_data_files(cfg.image_dir, cfg.label_dir, cfg.val_ids)

    cache_num_workers = 1 if cfg.crop_lung else cfg.num_workers
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=cache_num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=cache_num_workers)
    loader_num_workers = cfg.num_workers
    if is_debugging():
        logging.debug("Data loader num_workers set to 0 for debugging.")
        loader_num_workers = 0  # debugging is difficult when multiple processes are running
    batch_size = cfg.batch_size // cfg.patch_per_sample
    if batch_size > len(train_ds):
        logging.warning(f"Effective batch-size ({batch_size}) is larger than the number of training samples.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=loader_num_workers)

    return train_loader, val_loader


def train_batch(batch_data, model, criterion, optimizer, device):
    model.train()
    start = time.time()

    inputs, labels = (
        batch_data["image"].to(device),
        batch_data["label"].to(device),
    )
    # ------------------ forward -------------------
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # ------------------ backward ------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # --------------- adjusting lr -----------------
    lr = optimizer.param_groups[0]["lr"]
    # lr_scheduler.step()

    if device.type == "cuda":
        torch.cuda.synchronize()  # in order to calculate time correctly
    end = time.time()
    return loss.item(), lr, end - start


@torch.no_grad()
def validate(val_loader, model, sw_size, sw_batch_size, device, debug_output=None):
    model.eval()
    start = time.time()

    # -------- building transforms and metric ---------
    if debug_output:
        save_transform = SaveImaged(keys=["label", "pred"], resample=False, output_postfix="",
                                    separate_folder=False, output_dir=debug_output)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    # ---------------- validation loop -----------------
    for val_data in val_loader:
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )
        # --------------- predicting masks ----------------
        val_outputs = sliding_window_inference(val_inputs, sw_size, sw_batch_size, model)
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        # ----- computing dice for current iteration ------
        dice_metric(y_pred=val_outputs, y=val_labels)
        # ----- saving predicted masks for debugging ------
        if debug_output:
            for val_output, val_d in zip(val_outputs, decollate_batch(val_data)):
                add_new_key(val_d, "pred", val_output[1:2].cpu())
                save_transform(val_d)
    # --- aggregating the final mean dice result ---
    dice = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()

    end = time.time()
    return dice, end - start


def main(cfg, debug=False):
    os.makedirs(cfg.run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, "tensorboard"))
    init_log(log_path=os.path.join(cfg.run_dir, "training.log"), debug=debug)
    log_info(cfg)

    # set deterministic training for reproducibility
    set_determinism(seed=cfg.seed)

    device = torch.device("cpu")
    if cfg.gpu_id is not None:
        device = torch.device("cuda")
        torch.cuda.set_device(cfg.gpu_id)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # due to performance issues

    # ------------- building data loaders -------------
    train_loader, val_loader = build_loaders(cfg)

    debug_output = ""
    if debug:
        check_transform(train_loader.dataset, os.path.join(cfg.run_dir, "samples", "train"))
        check_transform(val_loader.dataset, os.path.join(cfg.run_dir, "samples", "val"))
        debug_output = os.path.join(cfg.run_dir, "results/{}")

    # ---- building model, criterion and optimizer ----
    weights = None
    if cfg.get("pretrained_weights", None) is not None:
        logging.info(f"Loading pretrained weights '{cfg.pretrained_weights}'.")
        weights = torch.load(cfg.pretrained_weights)
    model = build_model(cfg.model, weights).to(device)
    criterion = build_loss(cfg.loss).to(device)
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr)

    # ---------------- training loop -----------------
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    best_dice = 0
    best_iter = 0
    start = end = time.time()
    for step, batch_data in enumerate(cycle(train_loader, stop=cfg.iterations), start=1):
        num_samples = batch_data["image"].size(0)
        data_time.update(time.time() - end, num_samples)
        loss, lr, train_batch_time = train_batch(batch_data, model, criterion, optimizer, device)
        loss_meter.update(loss, num_samples)
        batch_time.update(train_batch_time, num_samples)
        # ------------- logging train status ------------
        if step % cfg.log_freq == 0:
            logging.info(f"Train: [{step}/{cfg.iterations}]  \t" +
                         f"Data-Time {data_time.avg:.3f}   " +
                         f"Batch-Time {batch_time.avg:.3f}   " +
                         f"Loss {loss_meter.avg:.5f}   " +
                         f"Learning-Rate {lr:.6f}")
            writer.add_scalar("train/loss", loss_meter.avg, step)
            writer.add_scalar("train/learning_rate", lr, step)
            batch_time.reset()
            data_time.reset()
            loss_meter.reset()
        # -------------- validating model ---------------
        if step % cfg.val_freq == 0:
            logging.info("-" * 40)
            dice, val_total_time = validate(val_loader, model, cfg.sliding_window_size, cfg.sliding_window_batch_size,
                                            device, debug_output.format(step))
            # --------- logging validation result ----------
            logging.info(f"Validation: [{step}/{cfg.iterations}]  \t" +
                         f"Total-Time {val_total_time:.3f}   " +
                         f"Mean-Dice {dice:.4f}")
            writer.add_scalar("valid/mean-dice", dice, step)
            # ----------- saving best checkpoint -----------
            if dice > best_dice:
                best_dice = dice
                best_iter = step
                save_checkpoint(cfg, model, step, dice)
            logging.info(f"Best Mean-Dice {best_dice:.4f} @ Iteration {best_iter}")
            logging.info("-" * 40)
        end = time.time()
    logging.info(f"End of training, Total-Time {end - start:.2f}, " +
                 f"Best Mean-Dice {best_dice:.4f} @ Iteration {best_iter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung Segmentation Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    args = parser.parse_args()
    cfg = get_config(args.config)
    main(cfg, args.debug)
