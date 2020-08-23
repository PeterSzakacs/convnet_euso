import os
import random
import unittest
import uuid

import numpy as np
import numpy.testing as nptest

import dataset.io.fs.facades.test.test_base as test_base
import dataset.io.fs.test.numpy_utils as numpy_utils


class BaseMemmapFacadeTest(test_base.BaseFacadeTest):

    @classmethod
    def setUpClass(cls):
        super(BaseMemmapFacadeTest, cls).setUpClass()
        cls._utils = numpy_utils.NumpyTestUtils()

    @classmethod
    def _get_facade_key(cls):
        return 'memmap'

    def _create_memmap(self, filename, shape, dtype):
        filepath = os.path.join(self._temp_dir, filename)
        return filepath, self._utils.create_memmap(filepath,
                                                   shape=shape,
                                                   dtype=dtype)


class TestLoadData(BaseMemmapFacadeTest):

    def test_load(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        num_items, item_shape = shape[0], shape[1:]

        # first create a temporary memmap on the filesystem
        tmpfile, mmap = self._create_memmap(f'test_load_{uuid.uuid4()}.memmap',
                                            shape=shape, dtype=dtype)
        mmap[:] = random.randint(1, 10)

        # try to load the data from the memmap and verify they match
        mmap2 = self._facade.load(tmpfile, item_shape=item_shape, dtype=dtype,
                                  num_items=num_items)
        nptest.assert_array_equal(mmap, mmap2)
        self.assertIsInstance(mmap2, np.memmap)

    def test_load_nonexistent_memmap(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        num_items, item_shape = shape[0], shape[1:]

        # try to load the data from the memmap and verify they match
        filename = 'test_load_not_exists.memmap'
        tmpfile = os.path.join(self._temp_dir, filename)
        self.assertRaises(FileNotFoundError, self._facade.load,
                          filename=tmpfile, item_shape=item_shape,
                          dtype=dtype, num_items=num_items)


class TestSaveData(BaseMemmapFacadeTest):

    def test_save_memmap(self):
        dtype, shape = self._utils.get_random_ndarray_params()

        # first create a temporary memmap on the filesystem
        file1, arr = self._create_memmap(f'test_save_{uuid.uuid4()}_1.memmap',
                                         shape=shape, dtype=dtype)
        arr[:] = random.randint(1, 10)

        # persist the memmap to a new file and verify that attempting to read
        # it back from the new location returns the same data
        file2 = f'test_save_{uuid.uuid4()}_2.memmap'
        file2 = os.path.join(self._temp_dir, file2)
        self._facade.save(file2, arr)

        self.assertTrue(os.path.isfile(file2))
        mmap = np.memmap(file2, dtype=dtype, shape=shape, mode='r')
        nptest.assert_array_equal(mmap, arr)

    def test_save_ndarray(self):
        dtype, shape = self._utils.get_random_ndarray_params()

        # create the array to save
        arr = np.zeros(shape=shape, dtype=dtype)

        # persist the ndarray and verify that attempting to read it back from
        # the filesystem returns the correct data
        _file = f'test_save_{uuid.uuid4()}.memmap'
        _file = os.path.join(self._temp_dir, _file)
        self._facade.save(_file, arr)

        self.assertTrue(os.path.isfile(_file))
        mmap = np.memmap(_file, dtype=dtype, shape=shape, mode='r')
        nptest.assert_array_equal(mmap, arr)


class TestAppendData(BaseMemmapFacadeTest):

    # positive test cases

    def test_append_ndarray_to_memmap(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        num_items, item_shape = shape[0], shape[1:]

        # first create a temporary memmap on the filesystem
        _file, mmap = self._create_memmap(f'test_append_{uuid.uuid4()}.memmap',
                                          shape=shape, dtype=dtype)
        mmap[:] = random.randint(1, 10)

        # create the array to append to the memmap
        arr = np.zeros(shape=shape, dtype=dtype)

        # append ndarray to memmap and verify that attempting to afterward read
        # back the memmap returns the correct data
        self._facade.append(_file, arr, dtype=dtype, num_items=num_items,
                            item_shape=item_shape)

        exp_shape = (2 * num_items, *shape[1:])
        result = np.memmap(_file, shape=exp_shape, dtype=dtype, mode='r')
        self.assertEqual(result.dtype, dtype)
        self.assertTupleEqual(result.shape, exp_shape)
        self.assertIsInstance(result, np.memmap)
        nptest.assert_array_equal(result[:num_items], mmap[:num_items])
        nptest.assert_array_equal(result[num_items:], arr)

    def test_append_memmap_to_memmap(self):
        dtype, shape = self._utils.get_random_ndarray_params()
        num_items, item_shape = shape[0], shape[1:]
        t = uuid.uuid4()

        # first create the memmap to append to
        file1, mmap1 = self._create_memmap(f'test_append_{t}_1.mmap',
                                           shape=shape, dtype=dtype)
        mmap1[:] = random.randint(1, 10)

        # create the memmap holding new data to be appended
        file2, mmap2 = self._create_memmap(f'test_append_{t}_2.mmap',
                                           shape=shape, dtype=dtype)
        mmap2[:] = random.randint(1, 10)

        # append second memmap to first and verify that attempting to afterward
        # read back the first memmap returns the concatenated data
        self._facade.append(file1, mmap2, dtype=dtype, num_items=num_items,
                            item_shape=item_shape)

        exp_shape = (2 * num_items, *shape[1:])
        result = np.memmap(file1, shape=exp_shape, dtype=dtype, mode='r')
        self.assertEqual(result.dtype, dtype)
        self.assertTupleEqual(result.shape, exp_shape)
        self.assertIsInstance(result, np.memmap)
        nptest.assert_array_equal(result[:num_items], mmap1[:num_items])
        nptest.assert_array_equal(result[num_items:], mmap2)

    # array/memmap parameter mismatch test cases

    def test_append_ndarray_wrong_dtype(self):
        dtype1, dtype2 = self._get_mismatched_dtypes()
        shape = self._utils.get_random_shape()
        num_items, item_shape = shape[0], shape[1:]

        # first create the memmap to append to
        file1, mmap1 = self._create_memmap(f'test_append_{uuid.uuid4()}_1.mmp',
                                           shape=shape, dtype=dtype1)
        mmap1[:] = random.randint(1, 10)

        # create the array to append
        arr = np.zeros(shape=shape, dtype=dtype2)

        # try to append and test for expected exception
        self.assertRaises(ValueError, self._facade.append, file1, arr,
                          dtype=dtype1, num_items=num_items,
                          item_shape=item_shape)

    def test_append_ndarray_wrong_item_shape(self):
        dtype = self._utils.get_random_dtype()

        shape1, shape2 = self._get_mismatched_shapes()
        num_items1, item_shape1 = shape1[0], shape1[1:]

        # first create the memmap to append to
        file1, mmap1 = self._create_memmap(f'test_append_{uuid.uuid4()}_1.map',
                                           shape=shape1, dtype=dtype)
        mmap1[:] = random.randint(1, 10)

        # create the array to append
        arr = np.zeros(shape=shape2, dtype=dtype)

        # try to append and test for expected exception
        self.assertRaises(ValueError, self._facade.append, file1, arr,
                          dtype=dtype, num_items=num_items1,
                          item_shape=item_shape1)

    def test_append_memmap_wrong_dtype(self):
        dtype1, dtype2 = self._get_mismatched_dtypes()
        shape = self._utils.get_random_shape()
        num_items, item_shape = shape[0], shape[1:]
        t = uuid.uuid4()

        # first create the memmap to append to
        file1, mmap1 = self._create_memmap(f'test_append_{t}_1.memmap',
                                           shape=shape, dtype=dtype1)
        mmap1[:] = random.randint(1, 10)

        # create the memmap holding new data to be appended
        file2, mmap2 = self._create_memmap(f'test_append_{t}_2.memmap',
                                           shape=shape, dtype=dtype2)
        mmap2[:] = random.randint(1, 10)

        # try to append and test for expected exception
        self.assertRaises(ValueError, self._facade.append, file1, mmap2,
                          dtype=dtype1, num_items=num_items,
                          item_shape=item_shape)

    def test_append_memmap_wrong_item_shape(self):
        shape1, shape2 = self._get_mismatched_shapes()
        num_items1, item_shape1 = shape1[0], shape1[1:]
        dtype = self._utils.get_random_dtype()
        t = uuid.uuid4()

        # first create the memmap to append to
        file1, mmap1 = self._create_memmap(f'test_append_{t}_1.memmap',
                                           shape=shape1, dtype=dtype)
        mmap1[:] = random.randint(1, 10)

        # create the memmap holding new data to be appended
        file2, mmap2 = self._create_memmap(f'test_append_{t}_2.memmap',
                                           shape=shape2, dtype=dtype)
        mmap2[:] = random.randint(1, 10)

        # try to append and test for expected exception
        self.assertRaises(ValueError, self._facade.append, file1, mmap2,
                          dtype=dtype, num_items=num_items1,
                          item_shape=item_shape1)

    # misc. (helper methods)

    def _get_mismatched_shapes(self):
        shape1 = self._utils.get_random_shape()
        shape2 = self._utils.get_random_shape()
        while shape2 == shape1:
            shape2 = self._utils.get_random_shape()
        return shape1, shape2

    def _get_mismatched_dtypes(self):
        dtype1 = self._utils.get_random_dtype()
        dtype2 = self._utils.get_random_dtype()
        while dtype2 == dtype1:
            dtype2 = self._utils.get_random_dtype()
        return dtype1, dtype2


if __name__ == '__main__':
    unittest.main()
