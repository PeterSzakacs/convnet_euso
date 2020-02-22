import unittest

import numpy as np
import numpy.testing as nptest

import dataset.targets.facades as facades


class TestTargetsFacade(unittest.TestCase):

    # custom asserts

    def assertTargetsTupleEqual(self, targets, exp_targets):
        self.assertEqual(len(targets), len(exp_targets))
        for idx in range(len(exp_targets)):
            nptest.assert_array_equal(targets[idx], exp_targets[idx])

    def _format_target_error(self, target_type, val):
        return self._targets_err.format(target_type, val)

    def assertTargetsDictEqual(self, targets, exp_targets):
        self.assertSetEqual(set(targets.keys()), set(exp_targets.keys()))
        for target_type in exp_targets.keys():
            target = targets.get(target_type)
            exp_target = exp_targets[target_type]
            keys_err_msg = self._format_target_error(target_type, target)
            nptest.assert_array_equal(target, exp_target, err_msg=keys_err_msg)

    # test setup

    @classmethod
    def setUpClass(cls):
        num_targets = 8
        targets = {
            "softmax_class_value": np.tile([0, 1], (num_targets, 1)),
        }
        cls.num_targets = num_targets
        cls.targets = targets
        cls._targets_err = "Expected targets for type {}, got {}"

    # test facade properties

    def test_target_types(self):
        included_types = ('softmax_class_value', )
        targets = {k: self.targets[k] for k in included_types}
        exp_targets = {k: True for k in included_types}

        facade = facades.TargetsFacade(targets)
        self.assertDictEqual(facade.target_types, exp_targets)

    def test_length(self):
        included_types = ('softmax_class_value', )
        targets = {k: self.targets[k] for k in included_types}

        facade = facades.TargetsFacade(targets)
        self.assertEqual(len(facade), self.num_targets)

    # test target retrieval methods (no copy on read)

    def test_get_targets_as_dict(self):
        included_types = ('softmax_class_value', )
        targets = self.targets
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = {k: targets[k] for k in included_types}

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsDictEqual(facade.get_targets_as_dict(), exp_targets)

    def test_get_data_as_dict_with_slice(self):
        included_types = ('softmax_class_value',)
        targets = self.targets
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = {k: targets[k][0:2] for k in included_types}

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsDictEqual(facade.get_targets_as_dict(slice(0, 2)),
                                    exp_targets)

    def test_get_targets_as_dict_with_range(self):
        included_types = ('softmax_class_value',)
        targets = self.targets
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = {k: targets[k][range(1, 3)] for k in included_types}

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsDictEqual(facade.get_targets_as_dict(range(1, 3)),
                                    exp_targets)

    def test_get_targets_as_dict_with_indexes_list(self):
        included_types = ('softmax_class_value',)
        targets = self.targets
        indexes = [1, 4, 7]
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = {k: targets[k][indexes] for k in included_types}

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsDictEqual(facade.get_targets_as_dict(indexes),
                                    exp_targets)

    def test_get_targets_as_arraylike(self):
        included_types = ('softmax_class_value',)
        targets = self.targets
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = tuple(targets[k] for k in included_types)

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsTupleEqual(facade.get_targets_as_tuple(),
                                     exp_targets)

    def test_get_targets_as_arraylike_with_slice(self):
        included_types = ('softmax_class_value',)
        targets = self.targets
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = tuple(targets[k][4:6] for k in included_types)

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsTupleEqual(facade.get_targets_as_tuple(slice(4, 6)),
                                     exp_targets)

    def test_get_targets_as_arraylike_with_range(self):
        included_types = ('softmax_class_value',)
        targets = self.targets
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = tuple(targets[k][range(3, 7)] for k in included_types)

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsTupleEqual(facade.get_targets_as_tuple(range(3, 7)),
                                     exp_targets)

    def test_get_targets_as_arraylike_with_indexes_list(self):
        included_types = ('softmax_class_value',)
        targets = self.targets
        indexes = [1, 6, 3]
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = tuple(targets[k][indexes] for k in included_types)

        facade = facades.TargetsFacade(in_targets)
        self.assertTargetsTupleEqual(facade.get_targets_as_tuple(indexes),
                                     exp_targets)

    # test target shuffling

    def test_shuffle(self):
        def shuffler(seq):
            # set targets[0] to all zeroes and targets[3] to all sevens
            seq[0] = 0
            seq[3] = 7

        included_types = ('softmax_class_value',)
        targets = self.targets
        in_targets = {k: targets[k] for k in included_types}
        exp_targets = {k: targets[k].copy() for k in included_types}
        shuffler(exp_targets['softmax_class_value'])

        facade = facades.TargetsFacade(in_targets)
        facade.shuffle(shuffler, lambda: None)
        self.assertTargetsDictEqual(facade.get_targets_as_dict(), exp_targets)


if __name__ == '__main__':
    unittest.main()
