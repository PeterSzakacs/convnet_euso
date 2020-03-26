from . import memmap_io
from . import npy_io

HANDLERS = {
    'npy': npy_io.NumpyDataPersistencyHandler,
    'memmap': memmap_io.MemmapDataPersistencyHandler,
}
