import sys
import os
import numpy as np
import nibabel as nib

dataset_spacings = {
    "0202_001_V1_TLC": [0.76171899, 0.76171899, 0.625],
    "0202_001_V4_TLC": [0.76171899, 0.76171899, 0.625],
    "0496_202-032_V1_TLC": [0.61132813, 0.61132813, 0.6],
    "0496_202-032_V2_FRC": [0.67773438, 0.67773438, 0.6],
    "0496_202-032_V2_TLC": [0.67773438, 0.67773438, 0.6],
    "0499_502-2004_V2_TLC": [0.83593798, 0.83593798, 1.25],
    "0499_502-2004_V3_TLC": [0.80468798, 0.80468798, 1.25],
    "0499_502-2004_V4_TLC": [0.72070301, 0.72070301, 1.25],
    "0499_505-2004_V2_TLC": [0.73437500, 0.73437500, 1.25]
}


def get_spacing_from_affine(affine):
    RZS = affine[:3, :3]
    return np.sqrt(np.sum(RZS * RZS, axis=0))


def set_affine_spacing(affine, spacing):
    scale = np.divide(spacing, get_spacing_from_affine(affine))
    affine_transform = np.diag(np.ones(4))
    np.fill_diagonal(affine_transform, list(scale) + [1])
    return np.matmul(affine, affine_transform)


def get_nii_paths(dir_path):
    paths = []
    for (root, _, filenames) in os.walk(dir_path, topdown=False):
        for file in filenames:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                paths.append(os.path.join(root, file))
    return paths


if __name__ == "__main__":
    root_path = sys.argv[1]
    nii_paths = get_nii_paths(root_path)
    for counter, nii_file in enumerate(nii_paths):
        img_name = os.path.basename(nii_file).split('_pred')[0]
        img = nib.load(nii_file)
        img_spacing = get_spacing_from_affine(img.affine)
        new_spacing = np.array(dataset_spacings[img_name])
        if np.array_equal(np.around(img_spacing, 4), np.around(new_spacing, 4)):
            continue
        new_affine = set_affine_spacing(img.affine, new_spacing)
        nii_image = nib.Nifti1Image(img.get_fdata().astype(np.ushort), affine=new_affine)
        nib.save(nii_image, nii_file.replace("_pred", "_pred_"))
        print(f'{nii_file} is changed.')

    print('finished.')