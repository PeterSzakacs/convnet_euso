import unittest

from .. import get_facades_provider
from .. import base
from .. import memmap
from .. import npy


class TestGetFacadesProvider(unittest.TestCase):

    def test_get_numpy_facade(self):
        provider = get_facades_provider()
        facade = provider.get_instance('Npy')
        self.assertIsInstance(facade, npy.NumpyPersistenceFacade)

    def test_get_memmap_facade(self):
        provider = get_facades_provider()
        facade = provider.get_instance('MemMap')
        self.assertIsInstance(facade, memmap.MemMapFacade)

    def test_get_base_class(self):
        provider = get_facades_provider()
        self.assertIs(provider.base_class,
                      base.BaseFilesystemPersistenceFacade)

    def test_default_available_keys(self):
        provider = get_facades_provider()
        self.assertSetEqual(provider.available_keys,
                            {'npy', 'memmap'})


if __name__ == '__main__':
    unittest.main()
