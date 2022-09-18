from easydict import EasyDict

config = EasyDict()
config.seed = 13
config.run_dir = None
config.image_dir = "dataset/images"
config.label_dir = "dataset/labels/blood"
config.train_ids = [
    "0202_001_V1_TLC",
    "0202_001_V4_TLC",
    "0496_202-032_V1_TLC",
    "0499_502-2004_V2_TLC",
    "0499_502-2004_V3_TLC",
]
config.val_ids = [
    "0496_202-032_V2_TLC",
    "0499_502-2004_V4_TLC",
]
config.test_ids = [
    "0496_202-032_V2_FRC",
    "0499_505-2004_V2_TLC"
]
config.model = "UNet"
config.loss = "Dice"
config.optimizer = "Adam"
config.num_workers = 6
config.batch_size = 48
config.gpu_id = 0
config.lr = 1e-4
config.iterations = 5000
config.log_freq = 5
config.val_freq = 50
config.crop_lung = True
config.intensity_range = (-1000, 400)
config.spacing = (1.0, 1.0, 1.0)
config.patch_size = (96, 96, 96)
config.patch_per_sample = 16
config.pos_neg_ratio = 1
config.flip = False
config.rotate90 = False
config.shift_intensity = False
config.sliding_window_size = (160, 160, 160)
config.sliding_window_batch_size = 4
