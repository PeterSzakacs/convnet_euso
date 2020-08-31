import unittest

import dataset.io.fs.config.ini.base as base
import dataset.io.fs.config.ini.versions.version0 as v0


class TestModuleFunctions(unittest.TestCase):

    def test_get_config_parser_v0(self):
        parser = base.get_ini_parser(0)
        self.assertIsInstance(parser, v0.ConfigParser)


if __name__ == '__main__':
    unittest.main()
