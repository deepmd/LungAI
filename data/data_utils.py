import logging
import os

from monai.transforms import SaveImaged


def get_data_files(image_dir, label_dir, ids):
    image_dirs = sorted(x[0] for x in os.walk(image_dir) if not x[1] and x[0].endswith(tuple(ids)))
    image_files = image_dirs if os.listdir(image_dirs[0])[0].endswith(".dcm") else \
                  [os.path.join(x, os.listdir(x)[0]) for x in image_dirs]
    label_dirs = sorted(x[0] for x in os.walk(label_dir) if not x[1] and x[0].endswith(tuple(ids)))
    label_files = [(os.path.join(x, os.listdir(x)[0]) if len(os.listdir(x)) > 0 else None) for x in label_dirs]
    files = [
        ({"image": image_name, "label": label_name} if label_name is not None else {"image": image_name})
        for image_name, label_name in zip(image_files, label_files)
    ]
    return files


def check_transform(dataset, output_dir, num_samples=3):
    logging.debug(f"Saving some samples to '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)
    save_transform = SaveImaged(keys=["image", "label"], output_dir=output_dir, output_postfix="",
                                separate_folder=False, resample=False)
    for idx in range(min(len(dataset), num_samples)):
        check_data = dataset.__getitem__(idx)
        if isinstance(check_data, list):
            check_data = check_data[0]
        if idx == 0:
            image, label = (check_data["image"], check_data["label"])
            logging.debug(f"Image shape: {image.shape}, Label shape: {label.shape}")
        save_transform(check_data)
    logging.debug("=" * 40)