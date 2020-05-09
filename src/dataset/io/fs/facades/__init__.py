from dataset.io.fs.facades import memmap
from dataset.io.fs.facades import npy

IO_HANDLERS = {
    'npy': npy.NumpyPersistenceFacade(),
    'memmap': memmap.MemMapFacade(),
}
