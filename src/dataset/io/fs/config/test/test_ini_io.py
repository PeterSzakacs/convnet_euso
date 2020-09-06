import io
import os
import unittest
import unittest.mock as mock

import dataset.io.fs.config.ini.base as ini_base
import dataset.io.fs.config.ini_io as ini_io
import test.test_setups as testset


class TestIniConfigPersistenceManager(unittest.TestCase):

    class MockIniConverter(ini_base.AbstractIniConfigConverter):

        attrs = None

        @property
        def version(self):
            return 0

        def parse_config(self, raw_config):
            return {'test': 'val'}

        def create_config(self, dataset_attributes):
            self.attrs = dataset_attributes
            return {'general': dataset_attributes}

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    @mock.patch("os.path.isfile", return_value=True)
    def test_get_config_version_unversioned(self, m_isfile, m_open):
        m_open.return_value = io.StringIO("[general]\nnum_data = 0\n")

        manager = ini_io.IniConfigPersistenceManager()
        version = manager.get_config_version('test.ini')
        self.assertEqual(version, 0)

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    @mock.patch("os.path.isfile", return_value=True)
    def test_get_config_version_specific(self, m_isfile, m_open):
        m_open.return_value = io.StringIO("[general]\nversion = 2\n")

        manager = ini_io.IniConfigPersistenceManager()
        version = manager.get_config_version('test_config.ini')
        self.assertEqual(version, 2)

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    @mock.patch("os.path.isfile", return_value=True)
    def test_load_config(self, m_isfile, m_open):
        m_open.return_value = io.StringIO("[general]\nversion = 0\n")

        manager = ini_io.IniConfigPersistenceManager(
            config_converter=self.MockIniConverter())
        config = manager.load('test.ini')
        self.assertEqual(config['test'], 'val')

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    def test_save_config(self, m_open):
        buf = testset.MockTextFileStream()
        m_open.return_value = buf

        mock_converter = self.MockIniConverter()
        manager = ini_io.IniConfigPersistenceManager(
            config_converter=mock_converter)
        configfile = os.path.join('./test', 'test_config.ini')
        manager.save(configfile, {'version': 0})

        m_open.assert_called_with(configfile, 'w', encoding='UTF-8')
        self.assertEqual(f"[general]{os.linesep}version = 0{2 * os.linesep}",
                         buf.temp_buf)


if __name__ == '__main__':
    unittest.main()
