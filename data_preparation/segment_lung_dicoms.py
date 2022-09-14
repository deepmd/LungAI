import sys
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import pydicom as pd

from lungmask import mask


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


def get_dicom_paths(dir_path):
    paths = []
    for (root, _, filenames) in os.walk(dir_path, topdown=False):
        for file in filenames:
            if 'pickle' in file:
                file = file.replace('.pickle', '')
                paths.append(os.path.join(root, file))
    return paths


if __name__ == '__main__':
    root_dir = sys.argv[1]

    dicom_paths = get_dicom_paths(root_dir)
    model = mask.get_model(modeltype='unet', modelname='R231', modelpath="weights/unet_r231-d5d2fc3d.pth")

    for counter, dicoms_dir in enumerate(dicom_paths):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicoms_dir)
        reader.SetFileNames(dicom_names)
        input_image = reader.Execute()
        # size = input_image.GetSize()

        segmentation = mask.apply(input_image, model)
        # model prediction represents right lobe as 1 and left lobe as 2
        # we represent both of them as 1
        segmentation[segmentation > 0] = 1
        segmentation = segmentation.transpose((2, 1, 0))

        # save numpy as nifti
        out_path = dicoms_dir + '_lungmask.nii.gz'
        affine = create_affine(dicoms_dir)
        nii_image = nib.Nifti1Image(segmentation.astype(np.ushort), affine=affine)
        nib.save(nii_image, out_path)

        print(f'{counter + 1}/{len(dicom_paths)}: {out_path}')

    print('finished.')

