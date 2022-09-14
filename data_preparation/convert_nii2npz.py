import os
import sys

import nibabel as nib
import numpy as np


def get_dims(img):
    return (-img.affine[0,0], -img.affine[1,1], img.affine[2,2])


def get_nii_paths(dir_path):
    paths = []
    for (root, _, filenames) in os.walk(dir_path, topdown=False):
        for file in filenames:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                paths.append(os.path.join(root, file))
    return paths


if __name__ == '__main__':
    root_path = sys.argv[1]
    nii_paths = get_nii_paths(root_path)
    for counter, nii_path in enumerate(nii_paths):
        img = nib.load(nii_path)
        out_dir = os.path.dirname(nii_path)
        out_filename = os.path.basename(nii_path.replace(".nii.gz", ".npz").replace(".nii.gz", ".npz"))
        out_path = os.path.join(out_dir, "npz", out_filename)
        if os.path.exists(out_path):
            continue
        os.makedirs(os.path.join(out_dir, "npz"), exist_ok=True)
        dims = get_dims(img)
        img_array = np.array(img.dataobj).astype(bool)
        np.savez_compressed(out_path, dims=dims, mask=img_array)
        print(f'{counter+1}/{len(nii_paths)}: {nii_path}')

    print('finished.')