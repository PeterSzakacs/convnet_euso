import unittest

import numpy as np

import dataset.data.constants as cons
import dataset.data.utils as utils


class TestUtilsFunctions(unittest.TestCase):

    # test check item types

    def test_check_item_types(self):
        item_types = {k: False for k in cons.ALL_ITEM_TYPES}
        item_types['gtux'] = True
        try:
            utils.check_item_types(item_types)
        except ValueError:
            self.fail("No error expected to be raised, but ValueError was raised")

    def test_check_item_types_invalid_keys(self):
        # keys not defined in ALL_ITEM_TYPES are present
        item_types = {"test": True, "test2": True, "yx": True}
        self.assertRaises(ValueError, utils.check_item_types, item_types)

    def test_check_item_types_invalid_values(self):
        # values for all item types set to False
        item_types = {k: False for k in cons.ALL_ITEM_TYPES}
        self.assertRaises(ValueError, utils.check_item_types, item_types)

    # test check items length

    def test_check_items_length(self):
        items_length = 10
        items = {
            "yx": np.ones((items_length, 48, 48)),
            "gtux": np.ones((items_length, 20, 48))
        }
        try:
            utils.check_items_length(items)
        except ValueError:
            self.fail("No error expected to be raised, but ValueError was raised")

    def test_check_items_length_invalid(self):
        items_length = 10
        items = {
            "yx": np.ones((items_length, 48, 48)),
            "gtux": np.ones((items_length - 1, 20, 48))
        }
        self.assertRaises(ValueError, utils.check_items_length, items)
