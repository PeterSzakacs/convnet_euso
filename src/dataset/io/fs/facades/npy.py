import os
import typing as t

import numpy as np

from .base import BaseFilesystemPersistenceFacade


class NumpyPersistenceFacade(BaseFilesystemPersistenceFacade):
    """
    Facade providing functionality to manipulate dataset section items (only
    of a single item type) stored in numpy .npy files on disk.

    The keyword arguments of the to load() and append() methods allow to
    perform validation to ensure the parameters (dtype and actual shape) of
    the loaded ndarray match the passed in kwargs.
    """

    def load(
            self,
            filename: str,
            num_items: int = None,
            item_shape: t.Sequence[int] = None,
            dtype: t.Union[str, np.dtype] = None
    ):
        """
        Retrieve section items (of a single item type) stored in a numpy
        .npy file on disk.

        The keyword arguments to this function provide a type of validation.

        :param filename: Name/path of storing file
        :param dtype: (optional) the expected dtype of items in the loaded
                      array - if not set, will bypass checking
        :param num_items: (optional) expected number of items to load - if not
                          set, will bypass checking
        :param item_shape: (optional) expected shape of every item in the
                           retrieved array (i.e. every item gained by indexing
                           along axis 0) - if not set, will bypass checking
        :return: The retrieved items as a numpy.ndarray
        """
        items = np.load(filename)
        self._check_array_data(items, num_items, item_shape, dtype)
        return np.load(filename)

    def save(
            self,
            filename: str,
            items: np.ndarray
    ):
        """
        Persist passed in items (of a single item type) into a single numpy
        .npy file.

        :param filename: Name of new file to create
        :param items: The items to save
        """
        np.save(filename, items)

    def append(
            self,
            filename: str,
            items: np.ndarray,
            num_items: int = None,
            item_shape: t.Sequence[int] = None,
            dtype: t.Union[str, np.dtype] = None
    ):
        """
        Append passed in items (of a single item type) to already stored items
        of the same type in an existing numpy .npy file.

        Note that the already stored items can be optionally checked for
        expected dtype, number of stored items and shape of each item. In the
        case of the passed in new items, such checks would only be limited to
        dtype and item shape.

        :param filename: Name/path of file containing existing items
        :param items: The items to append
        :param num_items: (optional) the expected number of already persisted
                          items - if not set, will bypass checking
        :param item_shape: (optional) expected shape of every item (i.e. every
                           item gained by indexing along axis 0) - if not set,
                           will bypass checking
        :param dtype: (optional) the expected dtype of items - if not set, will
                      bypass checking
        """
        if dtype is not None and items.dtype != dtype:
            raise ValueError(f'Wrong array dtype. Expected {dtype}, '
                             f'but was {items.dtype}')
        if item_shape is not None and items.shape[1:] != item_shape:
            raise ValueError(f'Wrong item shape. Expected {item_shape}, '
                             f'but every item has shape {items.shape[1:]}')

        orig = np.load(filename, mmap_mode='r')
        self._check_array_data(orig, num_items, item_shape, dtype)

        arr = np.append(orig, items, axis=0)
        np.save(filename, arr)

    def delete(
            self,
            filename: str
    ):
        """
        Delete stored items (of a single item type) from the given file.

        Note that currently this method only removes any file located by its
        name, it performs no verification that the given file is a .npy file
        (in terms of its contents) or anything else.

        :param filename: Name/path of the file to delete
        """
        os.remove(filename)

    @staticmethod
    def _check_array_data(items, num_items, item_shape, dtype):
        if dtype is not None and items.dtype != dtype:
            raise ValueError(f'Wrong array dtype. Expected {dtype}, '
                             f'but was {items.dtype}')
        if num_items is not None and len(items) != num_items:
            raise ValueError(f'Wrong number of items. Expected {num_items}, '
                             f'but was {len(items)}')
        if item_shape is not None and items.shape[1:] != item_shape:
            raise ValueError(f'Wrong item shape. Expected {item_shape}, '
                             f'but every item has shape {items.shape[1:]}')
