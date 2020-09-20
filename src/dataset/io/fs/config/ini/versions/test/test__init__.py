import unittest

from .. import get_ini_converter


class TestPackageFunctions(unittest.TestCase):

    def test_get_config_converter_v0(self):
        from .. import version0 as v0
        parser = get_ini_converter(0)
        self.assertIsInstance(parser, v0.ConfigConverter)


if __name__ == '__main__':
    unittest.main()
