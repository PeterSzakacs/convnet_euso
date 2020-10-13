import unittest

import common.test_utils as tu
from .. import filename_utils as fu
from .. import get_formatters_provider


class TestGetFormattersProvider(unittest.TestCase):

    def test_get_base_class(self):
        provider = get_formatters_provider()
        self.assertEqual(provider.base_class, fu.FilenameFormatter)

    def test_get_default_available_formatters(self):
        provider = get_formatters_provider()
        exp_keys = {'type_only', 'name_with_type_suffix', 'name_with_suffix'}
        self.assertSetEqual(provider.available_keys, exp_keys)

    def test_get_name_only_formatter(self):
        format_name = tu.random_capitalize('type_only')
        provider = get_formatters_provider()
        formatter = provider.get_instance(format_name)
        self.assertIsInstance(formatter, fu.TypeOnlyFormatter)

    def test_get_name_with_type_suffix_formatter(self):
        format_name = tu.random_capitalize('name_with_type_suffix')
        provider = get_formatters_provider()
        formatter = provider.get_instance(format_name)
        self.assertIsInstance(formatter, fu.NameWithTypeSuffixFormatter)

    def test_get_name_with_suffix_formatter(self):
        format_name = tu.random_capitalize('name_with_suffix')
        provider = get_formatters_provider()
        formatter = provider.get_instance(format_name)
        self.assertIsInstance(formatter, fu.NameWithSuffixFormatter)


if __name__ == '__main__':
    unittest.main()
