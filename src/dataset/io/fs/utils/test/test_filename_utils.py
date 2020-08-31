import os
import unittest

# module functions to test
from ..filename_utils import append_file_extension, create_full_path
# classes to test
from ..filename_utils import TypeOnlyFormatter, NameWithSuffixFormatter, \
    NameWithTypeSuffixFormatter


class TestFileExtensionAppender(unittest.TestCase):

    def test_append_file_extension_single_filenme(self):
        expected = 'test_yx.npy'
        actual = append_file_extension("test_yx", 'npy')
        self.assertEqual(expected, actual)

    def test_append_file_extension_multiple_filenmes(self):
        filenames = ('test3_gtux', 'test1_yx', 't_raw')
        expected = ['test3_gtux.memmap', 'test1_yx.memmap', 't_raw.memmap']
        actual = list(append_file_extension(filenames, 'memmap'))
        self.assertListEqual(expected, actual)

    def test_append_file_extension_invalid_filenames(self):
        filenames = None
        self.assertRaises(ValueError, append_file_extension, filenames, 'npy')


class TestFullPathCreator(unittest.TestCase):

    def test_create_full_path_single_filenme(self):
        filename = 'file_raw'
        basedir = os.path.join('somedir', 'otherdir')

        expected = os.path.join(basedir, filename)
        actual = create_full_path(filename, basedir)
        self.assertEqual(expected, actual)

    def test_create_full_path_multiple_filenmes(self):
        filenames = ('test3_gtux.npy', 'test1_yx.memmap', 't_raw.npy')
        basedir = os.path.join('base', 'tmp')

        expected = [
            os.path.join(basedir, 'test3_gtux.npy'),
            os.path.join(basedir, 'test1_yx.memmap'),
            os.path.join(basedir, 't_raw.npy'),
        ]
        actual = list(create_full_path(filenames, basedir))
        self.assertListEqual(expected, actual)

    def test_create_full_path_invalid_filenames(self):
        filenames = None
        self.assertRaises(ValueError, create_full_path, filenames, 'tmp')


class TestTypeOnlyFormatter(unittest.TestCase):

    def test_get_single_file(self):
        formatter = TypeOnlyFormatter()
        item_type = 'yx'

        expected_filename = 'yx'
        actual_filename = formatter.create_filename(item_type)
        self.assertEqual(expected_filename, actual_filename)


class TestNameWithTypeSuffixFormatter(unittest.TestCase):

    def test_get_single_file(self):
        formatter = NameWithTypeSuffixFormatter()
        name, item_type = 'test_dataset', 'yx'

        expected_filename = 'test_dataset_yx'
        actual_filename = formatter.create_filename(name, item_type)
        self.assertEqual(expected_filename, actual_filename)

    def test_custom_delimiter(self):
        formatter = NameWithTypeSuffixFormatter()
        name, item_types = 'test', 'softmax_class_value'
        delimiter = '.'

        expected_filename = 'test.softmax_class_value'
        actual_filename = formatter.create_filename(name, item_types,
                                                    delimiter=delimiter)
        self.assertEqual(expected_filename, actual_filename)


class TestNameWithSuffixFormatter(unittest.TestCase):

    def test_get_single_file(self):
        formatter = NameWithSuffixFormatter()
        name, suffix = f'test_dataset', 'meta'

        expected_filename = 'test_dataset_meta'
        actual_filename = formatter.create_filename(name, suffix)
        self.assertEqual(expected_filename, actual_filename)

    def test_custom_delimiter(self):
        formatter = NameWithTypeSuffixFormatter()
        name, suffix = 'test', 'targets'
        delimiter = '.'

        expected_filename = 'test.targets'
        actual_filename = formatter.create_filename(name, suffix,
                                                    delimiter=delimiter)
        self.assertEqual(expected_filename, actual_filename)


if __name__ == '__main__':
    unittest.main()
