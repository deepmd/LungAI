import pickle
from os import PathLike
from typing import Sequence, Union

from monai.data import PydicomReader, is_supported_format
from monai.transforms import LoadImaged
from monai.utils import require_pkg, ensure_tuple


@require_pkg(pkg_name="pydicom")
class PicklePydicomReader(PydicomReader):
    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by PicklePydicom reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        suffixes: Sequence[str] = ["pickle"]
        return super().verify_suffix(filename) and is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.

        Args:
            data: file name or a list of file names to read,
            kwargs: do nothing

        Returns:
            If `data` represents a filename: return a list of pydicom dataset object.
            If `data` represents a list of filenames: return a list of list of pydicom dataset object.

        """
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        self.has_series = False

        for name in filenames:
            name = f"{name}"
            with open(name, "rb") as f:
                slices = pickle.load(f)
            slices = [ds for _, ds in sorted(slices.items(), key=lambda x: x[0])]
            img_.append(slices if len(slices) > 1 else slices[0])
            if len(slices) > 1:
                self.has_series = True
        return img_ if len(filenames) > 1 else img_[0]


class LoadImageAndPickled(LoadImaged):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register(PicklePydicomReader())