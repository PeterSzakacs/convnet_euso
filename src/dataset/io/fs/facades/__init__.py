from .base import BaseFilesystemPersistenceFacade
from .memmap import MemMapFacade
from .npy import NumpyPersistenceFacade

FACADES = {
    'npy': NumpyPersistenceFacade(),
    'memmap': MemMapFacade(),
}
