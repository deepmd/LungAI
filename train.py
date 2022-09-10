import argparse
import time

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from torch.utils.tensorboard import SummaryWriter

from utils import *


class Trainer:
    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        self.debug = debug
        os.makedirs(cfg.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, "tensorboard"))
        init_log(log_path=os.path.join(cfg.run_dir, "training.log"), debug=debug)
        log_info(cfg)

        # set deterministic training for reproducibility
        set_determinism(seed=cfg.seed)

        self.device = torch.device("cpu")
        if cfg.gpu_id is not None:
            self.device = torch.device("cuda")
            torch.cuda.set_device(cfg.gpu_id)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # due to performance issues

    @staticmethod
    def build_transforms(cfg):
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                # todo: crop lungs if required
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
                    pos=1,
                    neg=1,
                    num_samples=cfg.patch_per_sample,
                    image_key="image",
                    image_threshold=0,
                ),
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
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                # todo: crop lungs if required
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

    @staticmethod
    def check_transform(dataset, output_dir, num_samples=3):
        logging.debug(f"Saving some samples to '{output_dir}'")
        os.makedirs(output_dir, exist_ok=True)
        save_transform = SaveImaged(keys=["image", "label"], output_dir=output_dir, output_postfix="", resample=False)
        for idx in range(min(len(dataset), num_samples)):
            check_data = dataset.__getitem__(idx)
            if isinstance(check_data, list):
                check_data = check_data[0]
            if idx == 0:
                image, label = (check_data["image"], check_data["label"])
                logging.debug(f"Image shape: {image.shape}, Label shape: {label.shape}")
            save_transform(check_data)
        logging.debug("=" * 40)

    def _train(self, step, batch_data):
        self.model.train()
        start = time.time()
        inputs, labels = (
            batch_data["image"].to(self.device),
            batch_data["label"].to(self.device),
        )
        # ------------------ forward -------------------
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        # ------------------ backward ------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------ meters --------------------
        self.loss_meter.update(loss.item(), inputs.size(0))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.batch_time.update(time.time() - start, inputs.size(0))
        # ----------------- logging --------------------
        if step % self.cfg.log_freq == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            logging.info(f"Train: [{step}/{self.cfg.iterations}]  \t" +
                         f"Data-Time {self.data_time.avg:.3f}   " +
                         f"Batch-Time {self.batch_time.avg:.3f}   " +
                         f"Loss {self.loss_meter.avg:.5f}   " +
                         f"Learning-Rate {lr:.6f}")
            self.writer.add_scalar("train/loss", self.loss_meter.avg, step)
            self.writer.add_scalar("train/learning_rate", lr, step)
            self.batch_time.reset()
            self.data_time.reset()
            self.loss_meter.reset()
        # --------------- adjusting lr -----------------
        # self.lr_scheduler.step()

    @torch.no_grad()
    def _validate(self, step):
        self.model.eval()
        if self.debug:
            save_transform = SaveImaged(keys=["label", "pred"], resample=False, output_postfix="",
                                        output_dir=os.path.join(self.cfg.run_dir, f"results/{step:05d}"))
        start = time.time()
        # ---------------- validation loop -----------------
        for val_data in self.val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(self.device),
                val_data["label"].to(self.device),
            )
            # --------------- predicting masks ----------------
            roi_size = self.cfg.sliding_window_size
            sw_batch_size = self.cfg.sliding_window_batch_size
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, self.model)
            val_outputs = [self.post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [self.post_label(i) for i in decollate_batch(val_labels)]
            # ----- computing dice for current iteration ------
            self.dice_metric(y_pred=val_outputs, y=val_labels)
            # ----- saving predicted masks for debugging ------
            if self.debug:
                for val_output, val_d in zip(val_outputs, decollate_batch(val_data)):
                    add_new_key(val_d, "pred", val_output[1:2].cpu())
                    save_transform(val_d)
        # --- aggregating the final mean dice result ---
        metric = self.dice_metric.aggregate().item()
        # reset the status for next validation round
        self.dice_metric.reset()
        # ------------------ logging -------------------
        logging.info(f"Validation: [{step}/{self.cfg.iterations}]  \t" +
                     f"Total-Time {time.time() - start:.3f}   " +
                     f"Mean-Dice {metric:.4f}")
        self.writer.add_scalar("valid/mean-dice", metric, step)
        return metric

    def run(self):
        # ------------- building data loaders -------------
        train_transforms, val_transforms = self.build_transforms(self.cfg)
        train_files = get_data_files(self.cfg.image_dir, self.cfg.label_dir, self.cfg.train_ids)
        val_files = get_data_files(self.cfg.image_dir, self.cfg.label_dir, self.cfg.val_ids)

        train_ds = CacheDataset(data=train_files, transform=train_transforms,
                                cache_rate=1.0, num_workers=self.cfg.num_workers)
        val_ds = CacheDataset(data=val_files, transform=val_transforms,
                              cache_rate=1.0, num_workers=self.cfg.num_workers)
        loader_num_workers = self.cfg.num_workers
        if is_debugging():
            logging.debug("Data loader num_workers set to 0 for debugging.")
            loader_num_workers = 0
        batch_size = self.cfg.batch_size // self.cfg.patch_per_sample
        if batch_size > len(train_ds):
            logging.warning(f"Effective batch-size ({batch_size}) is larger than the number of training samples.")
        self.train_loader = DataLoader(train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=loader_num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=1, num_workers=loader_num_workers)

        if self.debug:
            self.check_transform(train_ds, os.path.join(self.cfg.run_dir, "samples", "train"))
            self.check_transform(val_ds, os.path.join(self.cfg.run_dir, "samples", "val"))

        # ----- building model, criterion and optimizer -----
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)
        self.criterion = DiceLoss(to_onehot_y=True, softmax=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])

        # ---------------- training loop -----------------
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter = AverageMeter()
        best_dice = 0
        best_iter = 0
        start = end = time.time()
        for step, batch_data in enumerate(cycle(self.train_loader, stop=self.cfg.iterations), start=1):
            self.data_time.update(time.time() - end, batch_data["image"].size(0))
            self._train(step, batch_data)
            if step % self.cfg.val_freq == 0:
                logging.info("-" * 40)
                dice = self._validate(step)
                if dice > best_dice:
                    best_dice = dice
                    best_iter = step
                    save_checkpoint(self.cfg, self.model, step, dice)
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
    trainer = Trainer(cfg, args.debug)
    trainer.run()
