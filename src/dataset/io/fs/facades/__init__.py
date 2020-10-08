import common.providers as providers

from .base import BaseFilesystemPersistenceFacade
from .memmap import MemMapFacade
from .npy import NumpyPersistenceFacade


def get_facades_provider() -> providers.ClassInstanceProvider:
    _facades_map = {
        'npy': {
            'class': NumpyPersistenceFacade,
        },
        'memmap': {
            'class': MemMapFacade,
        },
    }
    return providers.ClassInstanceProvider(
        class_map=_facades_map, base_class=BaseFilesystemPersistenceFacade
    )
