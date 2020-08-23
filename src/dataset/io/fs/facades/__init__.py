from .base import BaseFilesystemPersistenceFacade
from .memmap import MemMapFacade
from .npy import NumpyPersistenceFacade

IO_HANDLERS = {
    'npy': NumpyPersistenceFacade(),
    'memmap': MemMapFacade(),
}
