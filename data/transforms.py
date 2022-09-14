from typing import Dict, Hashable, Mapping

import numpy as np
import torch
from monai.utils.type_conversion import convert_to_tensor, convert_data_type
from monai.transforms.transform import MapTransform
from monai.transforms import MaskIntensity
from monai.config import KeysCollection

from lungmask import mask


class CropLungWithMaskd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            lung_key: str,
            allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.lung_key = lung_key

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        if self.lung_key in d.keys():
            for key in self.keys:
                if isinstance(d[key], torch.Tensor):
                    d[key] = torch.multiply(d[key], d[self.lung_key])
                elif isinstance(d[key], np.ndarray):
                    d[key] = np.multiply(d[key], d[self.lung_key])

            del d[self.lung_key]
            del d[f'{self.lung_key}_meta_dict']
        return d


class CropLungWithModeld(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 fill_value=-1200,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.model = mask.get_model(modeltype='unet', modelname='R231', modelpath="weights/unet_r231-d5d2fc3d.pth")
        self.fill_value = fill_value

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        assert len(self.keys) == 1, 'keys must only contain image key.'
        image_key = self.keys[0]
        image_np = convert_data_type(d[image_key], output_type=np.ndarray)[0]
        image_np = np.squeeze(image_np, axis=0)
        image_np = image_np.transpose((2, 1, 0))
        lung_mask = mask.apply(image_np, self.model)
        lung_mask[lung_mask > 1] = 1
        lung_mask = lung_mask.transpose((2, 1, 0))
        lung_mask = np.expand_dims(lung_mask, axis=0)
        lung_mask = convert_to_tensor(1-lung_mask, dtype=np.bool)
        d[image_key][lung_mask] = self.fill_value

        return d