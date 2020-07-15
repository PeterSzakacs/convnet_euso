from .base import FilesystemPersistenceFacade
from .memmap import MemMapFacade
from .npy import NumpyPersistenceFacade

IO_HANDLERS = {
    'npy': NumpyPersistenceFacade(),
    'memmap': MemMapFacade(),
}
