from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.nets import UNETR
from monai.networks.nets import SwinUNETR


def build_model(model_name, weights=None):
    if model_name.lower() == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    elif model_name.lower() == "unetr":
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    elif model_name.lower() == "swinunetr":
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
    else:
        raise ValueError(f"Model name '{model_name}' is not valid!")

    if weights is not None:
        if model_name.lower() == "swinunetr" and "state_dict" in weights:
            model.load_from(weights)
        else:
            model.load_state_dict(weights)
    return model
