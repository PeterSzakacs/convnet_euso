import unittest

import numpy.testing as nptest


class DataItemsDictUtilsMixin(unittest.TestCase):

    def setUp(self, items_error_template=None, dtype_error_template=None):
        self._items_err = (items_error_template
                           or "Expected items for item type {}, got {}")
        self._dtype_err = (dtype_error_template
                           or "Wrong data type for item '{}', expected {}, "
                              "got {}")

    def _format_item_error(self, item_type, val):
        return self._items_err.format(item_type, val)

    def _format_dtype_error(self, item_type, dtype, exp_dtype):
        return self._dtype_err.format(item_type, exp_dtype, dtype)

    def assertItemsDictEqual(self, items, exp_items):
        self.assertSetEqual(set(items.keys()), set(exp_items.keys()))
        for item_type in exp_items.keys():
            item, exp_item = items.get(item_type), exp_items[item_type]
            keys_err_msg = self._format_item_error(item_type, item)
            dtype_err_msg = self._format_dtype_error(item_type, item.dtype,
                                                     exp_item.dtype)
            nptest.assert_array_equal(item, exp_item, err_msg=keys_err_msg)
            self.assertEqual(item.dtype, exp_item.dtype, msg=dtype_err_msg)
