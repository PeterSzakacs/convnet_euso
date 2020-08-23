import os
import shutil
import typing as t

import numpy as np

from .base import BaseFilesystemPersistenceFacade


class MemMapFacade(BaseFilesystemPersistenceFacade):
    """
    Facade providing functionality to manipulate dataset section items (only
    of a single item type) stored in flat binary files which can be manipulated
    using the "memmap" functionality provided by Numpy.

    This is mainly to enable working with lightweight view-like objects when
    the data is too big to fit into main memory.
    """

    def load(
            self,
            filename: str,
            num_items: int,
            item_shape: t.Sequence[int],
            dtype: t.Union[str, np.dtype]
    ):
        """
        Retrieve section items (of a single item type) stored in a binary
        array on disk.

        :param filename: Name/path of storing file
        :param num_items: Number of items in the stored array
        :param item_shape: Shape of every item in the stored array (i.e. every
                           item gained by indexing along axis 0)
        :param dtype: Dtype of items in the stored array
        :return: The retrieved items as a numpy.memmap
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)

        mmap_shape = (num_items, *item_shape)
        _kwargs = {
            'offset': 0, 'order': 'C', 'mode': 'r',
            'shape': mmap_shape, 'dtype': dtype,
        }
        return np.memmap(filename, **_kwargs)

    def save(
            self,
            filename: str,
            items: np.ndarray,
    ):
        """
        Persist section items (of a single item type) to a binary array on
        disk.

        :param filename: Name/path of file to use for storing the items
        :param items: Items to store
        """
        dtype, mmap_shape = items.dtype, items.shape

        if isinstance(items, np.memmap):
            # if the array is a file-backed memmap already,
            # just copy its contents to the new file
            shutil.copyfile(items.filename, filename)
        else:
            # else create a new memmap with the given backing file
            _kwargs = {
                'offset': 0, 'order': 'C', 'mode': 'w+',
                'shape': mmap_shape, 'dtype': dtype,
            }
            mmap = np.memmap(filename, **_kwargs)
            mmap[:] = items[:]

    def append(
            self,
            filename: str,
            items: np.ndarray,
            num_items: int,
            item_shape: t.Sequence[int],
            dtype: t.Union[str, np.dtype]
    ):
        """
        Append section items (of a single item type) to already persisted items
        of the same type stored in a binary array on disk.

        The passed array of items to store is validated against the parameters
        of the already stored array (its items along axis 0 must have the same
        shape as the stored items and also the same dtype).

        :param filename: Name/path of storing file
        :param items: Items to store
        :param num_items: Number of items in the stored array
        :param item_shape: Shape of every item in the stored array (i.e. every
                           item gained by indexing along axis 0)
        :param dtype: Dtype of items in the stored array
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)

        self._check_array_data(items, item_shape, dtype)

        # calculate new shape of memmap after appending
        mmap_shape = (num_items + len(items), *item_shape)
        _kwargs = {
            'order': 'C', 'mode': 'r+',
            'shape': mmap_shape, 'dtype': dtype,
        }

        # create a memmap from the existing file (with new shape) and append
        # new data after the former end of the array data
        mmap = np.memmap(filename, **_kwargs)
        mmap[num_items:, :] = items

    def delete(
            self,
            filename: str
    ):
        """
        Delete stored items (of a single item type) from the given file.

        :param filename: Name/path of the file to delete
        """
        os.remove(filename)

    @staticmethod
    def _check_array_data(array, item_shape, dtype):
        if not array.shape[1:] == item_shape:
            raise ValueError(f'Wrong array item shape. Expected {item_shape}, '
                             f'but was {array.shape[1:]}')
        if not array.dtype == dtype:
            raise ValueError(f'Wrong array dtype. Expected {dtype}, '
                             f'but was {array.dtype}')
