import io
import unittest
import unittest.mock as mock

import test.test_setups as testset
import utils.io_utils as io_utils

@mock.patch('os.path.exists')
@mock.patch('os.path.isfile')
class TestModuleFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filename = 'test.tsv'
        cls.m_rows = [{'testkey': 'testvar', 'otherkey': 'otherval'}]
        cls.cols_order = ('testkey', 'otherkey')
        cls.file_contents = 'testkey\totherkey\r\ntestvar\totherval\r\n'

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Save_TSV(self, m_open, m_isfile, m_exists):
        m_exists.return_value = m_isfile.return_value = False
        buf = testset.MockTextFileStream()
        m_open.return_value = buf
        io_utils.save_TSV(self.filename, self.m_rows,
                          self.cols_order)
        m_open.assert_called_once_with(self.filename, 'w', encoding='UTF-8')
        self.assertEqual(buf.temp_buf, self.file_contents)

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Save_TSV_overwrite_existing_file(self, m_open, m_isfile,
                                              m_exists):
        m_exists.return_value = m_isfile.return_value = True
        buf = testset.MockTextFileStream(initial_value='foo')
        m_open.return_value = buf
        io_utils.save_TSV(self.filename, self.m_rows, self.cols_order,
                            file_exists_overwrite=True)
        m_open.assert_called_once_with(self.filename, 'w', encoding='UTF-8')
        self.assertEqual(buf.temp_buf, self.file_contents)

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Save_TSV_do_not_overwrite_existing_file(self, m_open, m_isfile,
                                                     m_exists):
        m_exists.return_value = m_isfile.return_value = True
        self.assertRaises(FileExistsError, io_utils.save_TSV, self.filename,
                          self.m_rows, self.cols_order,
                          file_exists_overwrite=False)
        m_open.assert_not_called()

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Load_TSV(self, m_open, m_isfile, m_exists):
        m_exists.return_value = m_isfile.return_value = True
        m_open.return_value = io.StringIO(self.file_contents)
        rows = io_utils.load_TSV(self.filename)
        self.assertEqual(rows, self.m_rows)
        m_open.assert_called_with(self.filename, 'r', encoding='UTF-8')


if __name__ == '__main__':
    unittest.main()
