import sys
import os
import numpy as np
import nibabel as nib


def create_affine(dims):
    """
    Function to generate the affine matrix for a dicom series
    This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)

    :param sorted_dicoms: list with sorted dicom files
    """

    image_orient1 = [1, 0, 0]
    image_orient2 = [0, 1, 0]
    delta_r = dims[0]
    delta_c = dims[1]
    step = [0, 0, dims[2]]
    image_pos = [0, 0, 0]

    affine = np.array(
        [[-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
         [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
         [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
         [0, 0, 0, 1]]
    )
    return affine


def get_npz_paths(dir_path):
    paths = []
    for (root, _, filenames) in os.walk(dir_path, topdown=False):
        for file in filenames:
            if '.npz' in file:
                paths.append(os.path.join(root, file))
    return paths


if __name__ == '__main__':
    root_path = sys.argv[1]
    npz_paths = get_npz_paths(root_path)
    # affine = np.array([[-1., 0., 0., 0.],
    #                    [0., -1., 0., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 0., 1.]])
    for counter, npz_path in enumerate(npz_paths):
        anno = np.load(npz_path)
        out_dir = os.path.dirname(npz_path)
        out_filename = os.path.splitext(os.path.basename(npz_path))[0] + ".nii.gz"
        out_path = os.path.join(out_dir, "nii", out_filename)
        if os.path.exists(out_path):
            continue
        os.makedirs(os.path.join(out_dir, "nii"), exist_ok=True)
        affine = create_affine(anno["dims"])
        nii_image = nib.Nifti1Image(anno["mask"].astype(np.ushort), affine=affine)
        nib.save(nii_image, out_path)
        print(f'{counter+1}/{len(npz_paths)}: {npz_path}')

    print('finished.')
