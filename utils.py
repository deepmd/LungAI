import datetime
import importlib
import inspect
import logging
import os
import sys

import torch
from monai.config import print_config, KeysCollection
from monai.utils import ensure_tuple


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_log(log_path, debug=False):
    log_root = logging.getLogger()
    log_root.setLevel((logging.INFO, logging.DEBUG)[debug or is_debugging()])
    handler_file = logging.FileHandler(log_path, mode="a")
    handler_stream = logging.StreamHandler(sys.stdout)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)


def log_info(cfg):
    logging.info(f"Start training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 40)
    logging.info("Parameters: ")
    cfg_dict = cfg.__dict__
    longest_key = max(len(k) for k in cfg_dict.keys())
    for name, value in cfg_dict.items():
        logging.info(f"{name.ljust(longest_key)} = {value}")
    logging.info("-" * 40)
    print_config(logging.root.handlers[0].stream)
    logging.info("=" * 40)


def save_checkpoint(cfg, model, step, dice):
    os.makedirs(os.path.join(cfg.run_dir, "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(cfg.run_dir, "checkpoints",
                                   f"ckpt_{step:05d}_dice{dice:.4f}.pth")
    logging.info(f"Saving checkpoint to {checkpoint_path}")
    state = {
        'config': cfg.__dict__,
        'state_dict': model.state_dict(),
        'iteration': step,
        'dice': dice
    }
    torch.save(state, checkpoint_path)


def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = os.path.basename(config_file)
    temp_module_name = os.path.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.%s" % temp_module_name)
    cfg = config.config
    if cfg.run_dir is None:
        cfg.run_dir = os.path.join('runs', temp_module_name)
    c = 1
    run_dir = cfg.run_dir
    while os.path.exists(run_dir):
        run_dir = cfg.run_dir + f" ({c})"
        c += 1
    cfg.run_dir = run_dir
    return cfg


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


def cycle(iterable, stop=sys.maxsize):
    c = 0
    while c < stop:
        for x in iterable:
            c += 1
            yield x


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def add_new_key(orig_data, key, new_data):
    orig_data[key] = new_data
    current_file_name = orig_data[key].meta["filename_or_obj"]
    base_name = current_file_name.rsplit('.')[0]
    exts = current_file_name.rsplit('.')[1:]
    orig_data[key].meta["filename_or_obj"] = '.'.join([base_name + f"_{key}"] + exts)


def from_engine(keys: KeysCollection, first: bool = False):
    keys = ensure_tuple(keys)

    def _wrapper(data):
        if isinstance(data, dict):
            return tuple(data[k] for k in keys)
        if isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [data[0][k] if first else [i[k] for i in data] for k in keys]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper
