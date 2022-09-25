import sys
import os
import nibabel as nib
import numpy as np


# def get_input_paths(*paths, flag):
#     if flag == "m":
#         file_name_parts = paths[1].split('_')
#         file_path = os.path.join(file_name_parts[0], '_'.join(file_name_parts[:-2]), 'nii')
#         image_names = paths[2].split(',')
#         paths = [os.path.join(paths[0], file_path, image_name.strip() + '.nii.gz') for image_name in image_names]
#     elif flag == "f":
#         file_name_parts = paths[1].split('_')
#         image_path = os.path.join(file_name_parts[0], '_'.join(file_name_parts[:-2]), 'nii')
#         lobe_path = os.path.join('lobe', file_name_parts[0], '_'.join(file_name_parts[:-2]))
#         image_names = paths[2].split(',')
#         paths = [os.path.join(paths[0], image_path, image_names[0].strip() + '.nii.gz'),
#                  os.path.join(paths[3], lobe_path, image_names[1].strip() + '.nii.gz')]
#     return paths

# def get_input_paths(root_path, file_name, image_names):
#     file_name_parts = file_name.split('_')
#     file_path = os.path.join(file_name_parts[0], '_'.join(file_name_parts[:-2]), 'nii')
#     image_names = image_names.split(',')
#     paths = [os.path.join(root_path, file_path, image_name.strip() + '.nii.gz') for image_name in image_names]
#     return paths


# def get_output_path(root_path, file_name):
#     file_name_parts = file_name.split('_')
#     file_path = os.path.join(file_name_parts[-2], file_name_parts[0], '_'.join(file_name_parts[:-2]))
#     path = os.path.join(root_path, file_path, file_name.strip() + '.nii.gz')
#     return path


def apply_merge(image_paths):
    mask = None
    for image_path in image_paths:
        image = nib.load(image_path)
        image_data = image.get_fdata()
        if mask is None:
            mask = np.ushort(image_data)
            affine = image.affine
        else:
            mask += np.ushort(image_data)

    return mask, affine, np.sum(np.unique(mask))


def apply_filter(image_paths):
    assert len(image_paths) == 2, 'For filtering, you must provide two images.'
    image1 = nib.load(image_paths[0])
    image1_data = np.ushort(image1.get_fdata())
    image2 = nib.load(image_paths[1])
    image2_data = np.ushort(image2.get_fdata())
    mask = np.multiply(image1_data, image2_data)
    return mask, image1.affine, np.sum(np.unique(mask))


def get_input_paths(root_dir, image_paths, out_dir=None, flag="m"):
    if flag == "m":
        image_paths = image_paths.split(',')
        paths = [os.path.join(root_dir, image_path.strip() + '.nii.gz') for image_path in image_paths]
    elif flag == "f":
        image_paths = image_paths.split(',')
        image_path = image_paths[0].strip()
        lobe_path = image_paths[1].strip()
        paths = [os.path.join(root_dir, image_path + '.nii.gz'),
                 os.path.join(out_dir, lobe_path + '.nii.gz')]
    return paths


if __name__ == "__main__":
    commands_file = sys.argv[1]
    input_dir = sys.argv[2]
    out_dir = sys.argv[3]

    with open(commands_file, 'r') as f:
        commands = f.readlines()

    for counter, command in enumerate(commands):
        command = command.strip('\n')
        if "-m" in command:
            mask_path, file_names = command.split("-m")
            file_paths = get_input_paths(input_dir, file_names, out_dir=None, flag="m")
            mask_data, affine, code = apply_merge(file_paths)
            mask_data[mask_data > 0] = 1
        elif "-f" in command:
            mask_path, file_names = command.split("-f")
            file_paths = get_input_paths(input_dir, file_names, out_dir, flag="f")
            mask_data, affine, code = apply_filter(file_paths)
            mask_data[mask_data > 0] = 1
        elif "-c" in command:
            mask_path, file_names = command.split("-c")
            file_paths = os.path.join(input_dir, file_names.strip() + '.nii.gz')
            mask = nib.load(file_paths)
            mask_data, affine = mask.get_fdata(), mask.affine
        else:
            raise ValueError(f'Command {command} is not valid.')

        out_path = os.path.join(out_dir, mask_path.strip() + '.nii.gz')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nii_image = nib.Nifti1Image(mask_data.astype(np.ushort), affine=affine)
        nib.save(nii_image, out_path)
        print(f'{counter+1}/{len(commands)}: {mask_path} is saved with code={code}.')

print("finished.")
