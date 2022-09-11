from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss, TverskyLoss


def build_loss(loss_name):
    if loss_name.lower() == "dice":
        loss = DiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name.lower() == "dicece":
        loss = DiceCELoss(to_onehot_y=True, softmax=True)
    elif loss_name.lower() == "dicefocal":
        loss = DiceFocalLoss(to_onehot_y=True, softmax=True)
    elif loss_name.lower() == "focal":
        loss = FocalLoss(to_onehot_y=True)
    elif loss_name.lower() == "tversky":
        loss = TverskyLoss(to_onehot_y=True, softmax=True)
    else:
        raise ValueError(f"Loss name '{loss_name}' is not valid!")

    return loss
