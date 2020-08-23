import random

import numpy as np


# Numpy utility functions for use in tests


class NumpyTestUtils:

    def __init__(
            self,
            min_ndims=2,
            max_ndims=4,
            min_dim_size=3,
            max_dim_size=10,
            dtypes_set=None
    ):
        if min_ndims > max_ndims:
            raise ValueError("Min dims must be less than or equal to max dims")
        if min_dim_size > max_dim_size:
            raise ValueError("Min dim size must be less than or equal to max "
                             "dim size")
        self._min_ndims = min_ndims
        self._max_ndims = max_ndims
        self._min_dim_size = min_dim_size
        self._max_dim_size = max_dim_size
        if not dtypes_set:
            sctypes = np.sctypes
            self._dtypes_set = list(set(
                sctypes['int'] + sctypes['uint'] + sctypes['float']
            ))
        else:
            self._dtypes_set = list(set(dtypes_set))

    def get_random_shape(self, ndims=None):
        randint = random.randint
        if ndims is None:
            ndims = randint(self._min_ndims, self._max_ndims)

        min_size, max_size = self._min_dim_size, self._max_dim_size

        # first dim is kept intentionally small to avoid large diff views
        # in assertion errors
        dims = [randint(1, 3)]
        for idx in range(1, ndims):
            dims.append(randint(min_size, max_size))
        return tuple(dims)

    def get_random_dtype(self):
        return random.choice(self._dtypes_set)

    def get_random_ndarray_params(self, ndims=None):
        dtype = self.get_random_dtype()
        shape = self.get_random_shape(ndims)
        return dtype, shape

    def create_memmap(self, filepath, shape=None, dtype=None, mode='w+',
                      scalar_fill_fn=None):
        if shape is None:
            shape = self.get_random_shape()
        if dtype is None:
            dtype = self.get_random_dtype()
        mmap = np.memmap(filepath, shape=shape, dtype=dtype, mode=mode)
        if scalar_fill_fn is not None:
            mmap[:] = scalar_fill_fn()
        return mmap

    def create_ndarray(self, shape=None, dtype=None, array_create_fn=None,
                       scalar_fill_fn=None):
        if shape is None:
            shape = self.get_random_shape()
        if dtype is None:
            dtype = self.get_random_dtype()
        if array_create_fn is None:
            array_create_fn = np.empty
        arr = array_create_fn(shape=shape, dtype=dtype)
        if scalar_fill_fn is not None:
            arr[:] = scalar_fill_fn()
        return arr
