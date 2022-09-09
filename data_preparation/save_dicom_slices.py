import sys
import os
import pickle


def get_pickle_paths(dir_path):
    paths = []
    for (root, _, filenames) in os.walk(dir_path, topdown=False):
        for file in filenames:
            if 'pickle' in file:
                paths.append(os.path.join(root, file))
    return paths


if __name__=='__main__':
    root_path = sys.argv[1]

    pickle_paths = get_pickle_paths(root_path)

    for counter, pickle_file in enumerate(pickle_paths):
        with open(pickle_file, 'rb') as pfile:
            slices = pickle.load(pfile)

        out_dir = pickle_file.rsplit('.')[0]
        os.makedirs(out_dir, exist_ok=True)
        for slice_name, slice in slices.items():
            slice_path = os.path.join(out_dir, slice_name + '.dcm')
            slice.save_as(slice_path)
        print(f'{counter+1}/{len(pickle_paths)}: {pickle_file}')

    print('finished.')