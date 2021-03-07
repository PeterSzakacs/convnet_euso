import unittest

from .. import base
from .. import get_layout_handlers_provider
from .. import handlers


class TestGetLayoutHandlersProvider(unittest.TestCase):

    def test_get_single_file_per_item_type_handler(self):
        provider = get_layout_handlers_provider()
        handler = provider.get_instance('Single_file_per_item_type')
        self.assertIsInstance(handler,
                              handlers.SingleFilePerItemTypeLayoutHandler)

    def test_get_base_class(self):
        provider = get_layout_handlers_provider()
        self.assertIs(provider.base_class,
                      base.DatasetSectionFileLayoutHandler)

    def test_default_available_keys(self):
        provider = get_layout_handlers_provider()
        self.assertSetEqual(provider.available_keys,
                            {'single_file_per_item_type'})


if __name__ == '__main__':
    unittest.main()
