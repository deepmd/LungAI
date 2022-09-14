import argparse
import time

from easydict import EasyDict
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    LabelToMaskd,
    Identity,
)
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, CacheDataset, Dataset, decollate_batch

from models import build_model
from data import LoadImageAndPickled, CropLungWithModeld
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

    evaluation_mode = all("label" in d for d in data_files)
    print(f"Starting {('inference', 'evaluation')[evaluation_mode]} using '{checkpoint_path}'")

    keys = ["image", "label"] if evaluation_mode else ["image"]
    test_transforms = Compose(
        [
            LoadImageAndPickled(keys=keys),
            EnsureChannelFirstd(keys=keys),
            (CropLungWithModeld(keys=["image"]) if cfg.get("crop_lung", False) else Identity()),
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
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    test_ds = Dataset(data=data_files, transform=test_transforms) if not cfg.get("crop_lung", False) else \
              CacheDataset(data=data_files, transform=test_transforms, cache_rate=1.0, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=(4, 0)[is_debugging()])

    post_transforms = Compose(
        [
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
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            (AsDiscreted(keys="label", to_onehot=2) if evaluation_mode else Identity())
        ]
    )
    save_transform = Compose(
        [
            LabelToMaskd(keys="pred", select_labels=1, merge_channels=True),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir,
                       output_postfix="pred", separate_folder=False, resample=False)
        ]
    )

    model = build_model(cfg.get("model", "UNet"), weights=checkpoint["state_dict"]).to(device)
    model.eval()
    start = time.time()
    # ----------- prediction/evaluation loop --------------
    for test_data in test_loader:
        test_inputs = test_data["image"].to(device)
        # --------------- predicting masks ----------------
        roi_size = cfg.sliding_window_size
        sw_batch_size = cfg.sliding_window_batch_size
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model).cpu()
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        save_transform(test_data)
        # ------------ computing dice score  --------------
        if evaluation_mode:
            test_outputs, test_labels = from_engine(["pred", "label"])(test_data)
            dice_metric(y_pred=test_outputs, y=test_labels)

    if evaluation_mode:
        metric = dice_metric.aggregate().item()
        print(f"Evaluation result:  \t" +
              f"Total-Time {time.time() - start:.3f}   " +
              f"Mean-Dice {metric:.4f}")
        return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung Segmentation Evaluation in Pytorch")
    parser.add_argument("checkpoint", type=str, help="path to the model checkpoint file")
    parser.add_argument("-o", "--out-dir", type=str, default=None, help="output path")
    parser.add_argument("-i", "--input-dir", type=str, default=None,
                        help="path of the input directory containing dicom series or pickle, nifti, ... files")
    parser.add_argument("-g", "--gpu-id", type=int, default=None, help="id of the gpu to be used")
    args = parser.parse_args()
    data_files = None
    if args.input_dir is not None:
        data_files = [{"image": args.input_dir}] if os.listdir(args.input_dir)[0].endswith(".dcm") else \
                     [{"image": os.path.join(args.input_dir, f)} for f in os.listdir(args.input_dir)]
    if not args.out_dir:
        args.out_dir = os.path.splitext(args.checkpoint)[0]
    os.makedirs(args.out_dir, exist_ok=True)
    eval(args.checkpoint, data_files, args.out_dir, args.gpu_id)
