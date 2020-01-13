import os
import re
import unittest
import unittest.mock as mock

import numpy.testing as nptest

import dataset.constants as cons
import dataset.io.fs.data.npy_io as npy_io
import test.test_setups as testset


class TestNumpyDataPersistencyManager(testset.DatasetItemsMixin,
                                      unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestNumpyDataPersistencyManager, cls).setUpClass()
        cls.name = 'test_name'
        cls.loaddir, cls.savedir = '/dsets', '/test'
        suffixes = {k: '_{}_test'.format(k) for k in cons.ALL_ITEM_TYPES}
        cls.datafiles = {k: '{}{}.npy'.format(cls.name, suffixes[k])
                         for k in cons.ALL_ITEM_TYPES}
        with mock.patch('os.path.isdir', return_value=True), \
             mock.patch('os.path.exists', return_value=True):
            cls.handler = npy_io.NumpyDataPersistencyHandler(
                cls.loaddir,
                cls.savedir,
                data_files_suffixes=suffixes
            )

    @mock.patch('numpy.load')
    def test_load_data(self, m_load):
        name, items, itypes = self.name, self.items, self.item_types
        pattern = '_(raw|gtux|gtuy|yx)_test.npy'
        i_getter = (lambda filename:
                    items[re.search(pattern, filename).group(1)])
        m_load.side_effect = i_getter
        exp_items = {k: ([] if not v else items[k]) for k, v in itypes.items()}

        dset_data = self.handler.load_data(name, itypes)
        self.assertDictEqual(dset_data, exp_items)

    @mock.patch('numpy.save')
    def test_save_data(self, m_save):
        name, items = self.name, self.items
        exp_filenames = {k: os.path.join(self.savedir, self.datafiles[k])
                         for k in items.keys()}

        filenames = self.handler.save_data(name, items)
        # assert save was not called for any type in item_types being False
        self.assertDictEqual(filenames, exp_filenames)
        self.assertEqual(m_save.call_count, len(items))
        pattern = '_(raw|gtux|gtuy|yx)_test.npy'
        for cal in m_save.call_args_list:
            filename = cal[0][0]
            itype = re.search(pattern, filename).group(1)
            self.assertEqual(filename, exp_filenames[itype])
            nptest.assert_array_equal(cal[0][1], items[itype])


if __name__ == '__main__':
    unittest.main()
