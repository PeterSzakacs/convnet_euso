import io
import os
import unittest
import unittest.mock as mock

import test.test_setups as testset
import dataset.io.fs.config.ini_io as ini_io
import dataset.io.fs.config.ini.base as ini_base


class TestIniConfigPersistencyHandler(unittest.TestCase):

    class MockIniParser(ini_base.AbstractIniConfigParser):

        attrs = None

        @property
        def version(self):
            return 0

        def parse_config(self, raw_config):
            return {'test': 'val'}

        def create_config(self, dataset_attributes):
            self.attrs = dataset_attributes
            return {'general': dataset_attributes}

    @classmethod
    def setUp(cls):
        pass

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("os.path.isfile", return_value=True)
    def test_get_config_version_unversioned(self, m_isfile, m_isdir, m_open):
        m_open.return_value = io.StringIO("[general]\nnum_data = 0\n")

        handler = ini_io.IniConfigPersistencyHandler(load_dir='./test')
        version = handler.get_config_version('test')
        self.assertEqual(version, 0)

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("os.path.isfile", return_value=True)
    def test_get_config_version_specific(self, m_isfile, m_isdir, m_open):
        m_open.return_value = io.StringIO("[general]\nversion = 2\n")

        handler = ini_io.IniConfigPersistencyHandler(load_dir='./test')
        version = handler.get_config_version('test')
        self.assertEqual(version, 2)

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("os.path.isfile", return_value=True)
    def test_load_config(self, m_isfile, m_isdir, m_open):
        m_open.return_value = io.StringIO("[general]\nversion = 0\n")

        handler = ini_io.IniConfigPersistencyHandler(
            config_parser=self.MockIniParser(), load_dir='./test')
        config = handler.load_config('test')
        print('test')
        # self.assertEqual(config, 0)

    @mock.patch("builtins.open", new_callable=mock.mock_open())
    @mock.patch("os.path.isdir", return_value=True)
    def test_save_config(self, m_isdir, m_open):
        buf = testset.MockTextFileStream()
        m_open.return_value = buf

        mock_parser = self.MockIniParser()
        handler = ini_io.IniConfigPersistencyHandler(
            config_parser=mock_parser, save_dir='./test')
        handler.save_config('test', {'version': 0})

        exp_configfile = os.path.join('./test', 'test_config.ini')
        m_open.assert_called_with(exp_configfile, 'w', encoding='UTF-8')
        self.assertEqual(f"[general]{os.linesep}version = 0{2 * os.linesep}",
                         buf.temp_buf)


if __name__ == '__main__':
    unittest.main()
