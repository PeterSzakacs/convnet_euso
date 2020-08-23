import os
import random
import unittest
import uuid

import numpy as np
import numpy.testing as nptest

import dataset.io.fs.facades.test.test_base as test_base
import dataset.io.fs.test.numpy_utils as numpy_utils


class BaseNumpyFacadeTest(test_base.BaseFacadeTest):

    @classmethod
    def setUpClass(cls):
        super(BaseNumpyFacadeTest, cls).setUpClass()
        cls._utils = numpy_utils.NumpyTestUtils()

    @classmethod
    def _get_facade_key(cls):
        return 'npy'


class TestLoadData(BaseNumpyFacadeTest):

    def test_load(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        num_items, item_shape = shape[0], shape[1:]

        # first create a .npy file on the filesystem
        _file = f'test_load_{uuid.uuid4()}.npy'
        _file = os.path.join(self._temp_dir, _file)
        arr1 = np.empty(shape=shape, dtype=dtype)
        arr1[:] = random.randint(1, 10)
        np.save(_file, arr1)

        # try to load the data from the file and verify they match
        arr2 = self._facade.load(_file, item_shape=item_shape, dtype=dtype,
                                 num_items=num_items)
        nptest.assert_array_equal(arr1, arr2)
        self.assertNotIsInstance(arr2, np.memmap)


class TestSaveData(BaseNumpyFacadeTest):

    def test_save_memmap(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        t = uuid.uuid4()

        # first create a memmap on the filesystem
        file1 = f'test_save_{t}_1.memmap'
        file1 = os.path.join(self._temp_dir, file1)
        mmap = np.memmap(file1, shape=shape, dtype=dtype, mode='w+')
        mmap[:] = random.randint(1, 10)

        # persist the memmap to a new npy file and verify that attempting to
        # read it back from the new location returns the same data
        file2 = f'test_save_{t}_2.npy'
        file2 = os.path.join(self._temp_dir, file2)
        self._facade.save(file2, mmap)

        self.assertTrue(os.path.isfile(file2))
        arr = np.load(file2)
        nptest.assert_array_equal(arr, mmap)

    def test_save_ndarray(self):
        dtype, shape = self._utils.get_random_ndarray_params()

        # create the array to save
        _file = f'test_save_{uuid.uuid4()}.npy'
        _file = os.path.join(self._temp_dir, _file)
        arr1 = np.empty(shape=shape, dtype=dtype)
        arr1[:] = random.randint(1, 10)

        # persist the ndarray and verify that attempting to read it back from
        # the filesystem returns the correct data
        self._facade.save(_file, arr1)

        self.assertTrue(os.path.isfile(_file))
        arr2 = np.load(_file)
        nptest.assert_array_equal(arr1, arr2)


class TestAppendData(BaseNumpyFacadeTest):

    def test_append_ndarray_to_npy(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        num_items, item_shape = shape[0], shape[1:]

        # first create the .npy file on the filesystem to append to
        _file = f'test_load_{uuid.uuid4()}.npy'
        _file = os.path.join(self._temp_dir, _file)
        arr1 = np.empty(shape=shape, dtype=dtype)
        arr1[:] = random.randint(1, 10)
        np.save(_file, arr1)

        # create the array to append
        arr2 = np.zeros(shape=shape, dtype=dtype)

        # append ndarray to npy and verify that attempting to afterward read
        # back the memmap returns the correct data
        self._facade.append(_file, arr2, dtype=dtype, num_items=num_items,
                            item_shape=item_shape)

        exp_shape = (2 * num_items, *shape[1:])
        result = np.load(_file)
        self.assertEqual(result.dtype, dtype)
        self.assertTupleEqual(result.shape, exp_shape)
        nptest.assert_array_equal(result[:num_items], arr1)
        nptest.assert_array_equal(result[num_items:], arr2)

    def test_append_memmap_to_npy(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        num_items, item_shape = shape[0], shape[1:]
        t = uuid.uuid4()

        # first create a temporary npy file on the filesystem
        file1 = f'test_load_{t}.npy'
        file1 = os.path.join(self._temp_dir, file1)
        arr1 = np.empty(shape=shape, dtype=dtype)
        arr1[:] = random.randint(1, 10)
        np.save(file1, arr1)

        # create the memmap holding new data to be appended
        file2 = f'test_append_{t}_2.memmap'
        file2 = os.path.join(self._temp_dir, file2)
        mmap2 = np.memmap(file2, shape=shape, dtype=dtype, mode='w+')
        mmap2[:] = random.randint(1, 10)

        # append second memmap to first and verify that attempting to afterward
        # read back the first memmap returns the concatenated data
        self._facade.append(file1, mmap2, dtype=dtype, num_items=num_items,
                            item_shape=item_shape)

        exp_shape = (2 * num_items, *shape[1:])
        result = np.load(file1)
        self.assertEqual(result.dtype, dtype)
        self.assertTupleEqual(result.shape, exp_shape)
        nptest.assert_array_equal(result[:num_items], arr1)
        nptest.assert_array_equal(result[num_items:], mmap2)


if __name__ == '__main__':
    unittest.main()
