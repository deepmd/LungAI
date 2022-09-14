import sys
import os
import numpy as np
import nibabel as nib
import pydicom as pd


def create_affine(dicoms_dir):
    """
    Function to generate the affine matrix for a dicom series
    This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)

    :param sorted_dicoms: list with sorted dicom files
    """

    dicoms_files = sorted(os.listdir(dicoms_dir))
    first_dicom = pd.dcmread(os.path.join(dicoms_dir, dicoms_files[0]))
    last_dicom = pd.dcmread(os.path.join(dicoms_dir, dicoms_files[-1]))
    dicoms_num = len(dicoms_files)

    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    image_orient1 = np.array(first_dicom.ImageOrientationPatient)[0:3]
    image_orient2 = np.array(first_dicom.ImageOrientationPatient)[3:6]

    delta_r = float(first_dicom.PixelSpacing[0])
    delta_c = float(first_dicom.PixelSpacing[1])

    image_pos = np.array(first_dicom.ImagePositionPatient)

    last_image_pos = np.array(last_dicom.ImagePositionPatient)

    if dicoms_num == 1:
        # Single slice
        slice_thickness = 1
        if "SliceThickness" in first_dicom:
            slice_thickness = first_dicom.SliceThickness
        step = - np.cross(image_orient1, image_orient2) * slice_thickness
    else:
        step = (image_pos - last_image_pos) / (1 - dicoms_num)

    # check if this is actually a volume and not all slices on the same location
    if np.linalg.norm(step) == 0.0:
        raise ValueError("NOT_A_VOLUME")

    affine = np.array(
        [[-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
         [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
         [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
         [0, 0, 0, 1]]
    )
    return affine


def get_dicoms_dir(anno_dir):
    for ignore_dir in ["Blood", "blood", "Emph", "emph", "Lobe", "lobe"]:
        anno_dir = anno_dir.replace(ignore_dir, "")
    p = anno_dir.replace("Annotations", "CTs").rstrip("/")
    dicoms_dir = os.path.join(os.path.split(p)[0], "DICOM", os.path.split(p)[1])
    return dicoms_dir


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
    for counter, npz_path in enumerate(npz_paths):
        anno = np.load(npz_path)
        out_dir = os.path.dirname(npz_path)
        out_filename = os.path.splitext(os.path.basename(npz_path))[0] + ".nii.gz"
        out_path = os.path.join(out_dir, "nii", out_filename)
        if os.path.exists(out_path):
            continue
        os.makedirs(os.path.join(out_dir, "nii"), exist_ok=True)
        dicoms_dir = get_dicoms_dir(out_dir)
        affine = create_affine(dicoms_dir)
        nii_image = nib.Nifti1Image(anno["mask"].astype(np.ushort), affine=affine)
        nib.save(nii_image, out_path)
        print(f'{counter+1}/{len(npz_paths)}: {npz_path}')

    print('finished.')
