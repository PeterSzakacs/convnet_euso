import shutil

import numpy as np

from .base import FilesystemPersistenceFacade


class MemMapFacade(FilesystemPersistenceFacade):

    def load(self, filename, **kwargs):
        num_items, shape = int(kwargs['num_items']), tuple(kwargs['shape'])
        mmap_shape = (num_items, *shape)
        _kwargs = {
            'offset': 0, 'order': 'C', 'mode': 'r',
            # mandatory params
            'shape': mmap_shape,
            'dtype': kwargs['dtype'],
        }
        return np.memmap(filename, **_kwargs)

    def save(self, filename, array, **kwargs):
        if isinstance(array, np.memmap):
            # if the array is a file-backed memmap already,
            # just copy its contents to the new file
            shutil.copyfile(array.filename, filename)
        else:
            # else create a new memmap with the given backing file
            _kwargs = {
                'offset': 0, 'order': 'C', 'mode': 'w+',
                'shape': array.shape,
            }
            mmap = np.memmap(filename, **_kwargs)
            mmap[:] = array[:]

    def append(self, filename, array, **kwargs):
        # calculate new shape of memmap after appending
        num_items, shape = int(kwargs['num_items']), tuple(kwargs['shape'])
        mmap_shape = (num_items + len(array), *shape)
        _kwargs = {
            'mode': 'r+',
            # mandatory params
            'dtype': kwargs['dtype'],
            'shape': mmap_shape,
        }

        # create a memmap from the existing file (with new shape) and append
        # new data after the former end of the array data
        mmap = np.memmap(filename, **_kwargs)
        mmap[num_items:, :] = array
