import numpy as np

from .base import FilesystemPersistenceFacade


class NumpyPersistenceFacade(FilesystemPersistenceFacade):

    def load(self, filename, **kwargs):
        return np.load(filename)

    def save(self, filename, array, **kwargs):
        np.save(filename, array)

    def append(self, filename, array, **kwargs):
        orig = np.load(filename, mmap_mode='r')
        arr = np.append(orig, array, axis=0)
        np.save(filename, arr)
