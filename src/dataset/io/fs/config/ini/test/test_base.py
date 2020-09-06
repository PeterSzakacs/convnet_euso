import unittest

import dataset.io.fs.config.ini.base as base
import dataset.io.fs.config.ini.versions.version0 as v0


class TestModuleFunctions(unittest.TestCase):

    def test_get_config_converter_v0(self):
        parser = base.get_ini_converter(0)
        self.assertIsInstance(parser, v0.ConfigConverter)


if __name__ == '__main__':
    unittest.main()
