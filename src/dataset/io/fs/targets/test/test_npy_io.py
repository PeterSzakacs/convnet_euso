import os
import unittest
import unittest.mock as mock

import numpy.testing as nptest

import dataset.io.fs.targets.npy_io as npy_io
import test.test_setups as testset


class TestNumpyTargetsFsPersistencyManager(testset.DatasetTargetsMixin,
                                           unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestNumpyTargetsFsPersistencyManager, cls).setUpClass()
        cls.name, cls.loaddir, cls.savedir = 'test', '/dsets', '/test'
        # for targets, having live numpy arrays is not necessary
        file_suffix = '_class_targets_test'
        cls.targetsfile = '{}{}.npy'.format(cls.name, file_suffix)
        with mock.patch('os.path.isdir', return_value=True), \
             mock.patch('os.path.exists', return_value=True):
            cls.handler = npy_io.NumpyTargetsPersistencyHandler(
                cls.loaddir,
                cls.savedir,
                classification_targets_file_suffix=file_suffix
            )

    @mock.patch('numpy.load')
    def test_load_dataset_targets(self, m_load):
        m_load.return_value = self.mock_targets
        exp_filename = os.path.join(self.loaddir, self.targetsfile)
        dset_targets = self.handler.load_targets(self.name)
        nptest.assert_array_equal(dset_targets, self.mock_targets)
        m_load.assert_called_with(exp_filename)

    @mock.patch('numpy.save')
    def test_save_dataset_targets(self, m_save):
        exp_filename = os.path.join(self.savedir, self.targetsfile)
        filename = self.handler.save_targets(self.name, self.mock_targets)
        self.assertEqual(filename, exp_filename)
        m_save.assert_called_once_with(exp_filename, self.mock_targets)


if __name__ == '__main__':
    unittest.main()
