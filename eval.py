import argparse
import time

from easydict import EasyDict
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch

from utils import *


@torch.no_grad()
def eval(checkpoint_path, data_files, output_dir, gpu_id=None):
    checkpoint = torch.load(checkpoint_path)
    cfg = EasyDict(checkpoint["config"])
    if data_files is None:
        data_files = get_data_files(cfg.image_dir, cfg.label_dir, cfg.test_ids)

    device = torch.device("cpu")
    if gpu_id is not None:
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_id)
        torch.backends.cudnn.benchmark = True

    inference_only = all("label" in d for d in data_files)
    print(f"Starting {('evaluation', 'inference')[inference_only]} using '{checkpoint_path}'")

    if inference_only:
        test_transforms = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                # todo: crop lungs if required
                ScaleIntensityRanged(
                    keys=["image"], a_min=cfg.intensity_range[0], a_max=cfg.intensity_range[1],
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                # todo: zero-mean if required
                CropForegroundd(keys=["image"], source_key="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=cfg.spacing),
            ]
        )
    else:
        test_transforms = Compose(
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
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

    test_ds = Dataset(data=data_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=(4, 0)[is_debugging()])

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="", resample=False),
    ])

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    start = time.time()
    # ----------- prediction/evaluation loop --------------
    for test_data in test_loader:
        if inference_only:
            test_inputs = test_data["image"].to(device)
        else:
            test_inputs, test_labels = (
                test_data["image"].to(device),
                test_data["label"].to(device),
            )
        # --------------- predicting masks ----------------
        roi_size = cfg.sliding_window_size
        sw_batch_size = cfg.sliding_window_batch_size
        test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_data["pred"] = test_outputs
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        # ------------ computing dice score  --------------
        if not inference_only:
            test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]
            dice_metric(y_pred=test_outputs, y=test_labels)

    if not inference_only:
        metric = dice_metric.aggregate().item()
        print(f"Evaluation result:  \t" +
              f"Total-Time {time.time() - start:.3f}   " +
              f"Mean-Dice {metric:.4f}")
        return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung Segmentation Evaluation in Pytorch")
    parser.add_argument("checkpoint", type=str, help="path to the model checkpoint file")
    parser.add_argument("-o", "--out-dir", type=str, default=None, help="output path")
    parser.add_argument("-i", "--dcm-dir", type=str, default=None, help="path of dicom directory to be segmented")
    parser.add_argument("-g", "--gpu-id", type=int, default=None, help="id of the gpu to be used")
    args = parser.parse_args()
    data_files = [{"image": args.dcm_dir}] if args.dcm_dir is not None else None
    if not args.out_dir:
        args.out_dir = os.path.splitext(args.checkpoint)[0]
    os.makedirs(args.out_dir, exist_ok=True)
    eval(args.checkpoint, data_files, args.out_dir, args.gpu_id)
