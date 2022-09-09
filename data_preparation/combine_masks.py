import sys
import os
import numpy as np
import nibabel as nib


if __name__ == '__main__':
    lobe_path = '/media/1TB/Datasets/Behrad/Annotations/0499/0499_502-2004_V4_TLC/lobe/nii/lobe_lobe_MaskFromLobes.nii.gz'
    emph_path = '/media/1TB/Datasets/Behrad/Annotations/0499/0499_502-2004_V4_TLC/emph/nii/0499_502-2004_V4_TLC_emph_new.nii.gz'
    out_path = '/media/1TB/Datasets/Behrad/Annotations/0499/0499_502-2004_V4_TLC/emph/nii/0499_502-2004_V4_TLC_emph_FilterByLobes.nii.gz'

    lobe_image = nib.load(lobe_path)
    lobe_data = np.ushort(lobe_image.get_fdata())
    lobe_affine = lobe_image.affine
    # print(lobe_data.shape)
    # print(lobe_affine)

    emph_image = nib.load(emph_path)
    emph_data = np.ushort(emph_image.get_fdata())
    emph_affine = emph_image.affine
    # print(emph_data.shape)
    # print(emph_affine)

    mask = np.multiply(lobe_data, emph_data)
    check = np.unique(mask)
    print(check)

    nii_image = nib.Nifti1Image(mask.astype(np.ushort), affine=emph_affine)
    nib.save(nii_image, out_path)




