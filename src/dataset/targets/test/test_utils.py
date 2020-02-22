import unittest

import numpy as np

import dataset.targets.constants as cons
import dataset.targets.utils as utils


class TestUtilsFunctions(unittest.TestCase):

    # test check item types

    def test_check_item_types(self):
        item_types = {k: False for k in cons.TARGET_TYPES}
        item_types['softmax_class_value'] = True
        try:
            utils.check_target_types(item_types)
        except ValueError:
            self.fail("No error expected to be raised, but ValueError was raised")

    def test_check_item_types_invalid_keys(self):
        # keys not defined in TARGET_TYPES are present
        item_types = {"test": True, "test2": True, "yx": True}
        self.assertRaises(ValueError, utils.check_target_types, item_types)

    def test_check_item_types_invalid_values(self):
        # values for all target types set to False
        item_types = {k: False for k in cons.TARGET_TYPES}
        self.assertRaises(ValueError, utils.check_target_types, item_types)

    # test check targets length

    def test_check_targets_length(self):
        items_length = 10
        items = {
            "softmax_class_value": np.tile([[0, 1], [1, 0]], (items_length, 1)),
        }
        try:
            utils.check_targets_length(items)
        except ValueError:
            self.fail("No error expected to be raised, but ValueError was raised")

    def test_check_targets_length_invalid(self):
        items_length = 10
        items = {
            "softmax_class_value": np.tile([[1, 0], [0, 1]], (items_length, 1)),
            "test": [0] * (items_length + 1)
            # test is not a defined target type, but this function should not
            # even check valid target types
        }
        self.assertRaises(ValueError, utils.check_targets_length, items)
