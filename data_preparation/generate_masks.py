import sys
import os
import numpy as np
import nibabel as nib


def get_nii_paths(dir_path):
    paths = []
    for (root, _, filenames) in os.walk(dir_path, topdown=False):
        for file in filenames:
            if '.nii.gz' in file and '0499_505-2004_V2_TLC' in file:
                pos = file.split('.')[0].split('_')[-1]
                if pos in ['LLL', 'LUL', 'RLL', 'RML', 'RUL']:
                        paths.append(os.path.join(root, file))

    nii_paths = {'blood': [], 'emph': [], 'lobe': []}
    for path in paths:
        if 'blood' in path:
            nii_paths['blood'].append(path)
        elif 'emph' in path:
            nii_paths['emph'].append(path)
        else:
            nii_paths['lobe'].append(path)

    return nii_paths


if __name__ == '__main__':
    root_dir = sys.argv[1]

    nii_paths = get_nii_paths(root_dir)

    for key in ['lobe']:
        sum_image = None
        for nii_path in nii_paths[key]:
            image = nib.load(nii_path)
            image_data = image.get_fdata()
            if sum_image is None:
                sum_image = np.ushort(image_data)
                affine = image.affine
            else:
                sum_image += np.ushort(image_data)
        check = np.unique(sum_image)
        print(check)
        filename = nii_paths[key][0].split('/')[-3] + f'_{key}_MaskFromLobes.nii.gz'
        out_path = os.path.dirname(nii_paths[key][0])
        out_path = os.path.join(out_path, filename)
        nii_image = nib.Nifti1Image(sum_image.astype(np.ushort), affine=affine)
        nib.save(nii_image, out_path)

    print('finished.')
