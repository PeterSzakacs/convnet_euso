import unittest

from .. import get_config_manager
from ..ini import managers


class TestPackageFunctions(unittest.TestCase):

    def test_get_default_config_manager(self):
        manager = get_config_manager()
        self.assertIsInstance(manager, managers.IniConfigPersistenceManager)

    def test_get_ini_config_manager(self):
        manager = get_config_manager(config_format='INI')
        self.assertIsInstance(manager, managers.IniConfigPersistenceManager)


if __name__ == '__main__':
    unittest.main()
