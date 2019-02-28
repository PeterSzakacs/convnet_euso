import unittest

import test.test_setups as setups
import utils.network_utils as netutils


class TestDatasetSplitter(setups.DatasetMixin, unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        super(TestDatasetSplitter, cls).setUpClass(num_items=10, item_types={
            'yx': True, 'gtux': False, 'gtuy': True, 'raw': False})

    # test methods

    def test_get_indices_with_split_mode_from_start_with_num(self):
        num_items = 3
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            num_items=num_items)
        train, test = splitter.get_train_test_indices(10)
        self.assertListEqual(train, [3, 4, 5, 6, 7, 8, 9])
        self.assertListEqual(test, [0, 1, 2])

    def test_get_indices_with_split_mode_from_start_with_fraction(self):
        frac = 0.4
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            items_fraction=frac)
        train, test = splitter.get_train_test_indices(10)
        self.assertListEqual(train, [4, 5, 6, 7, 8, 9])
        self.assertListEqual(test, [0, 1, 2, 3])

    def test_get_indices_with_split_mode_from_end_with_num(self):
        num_items = 4
        splitter = netutils.DatasetSplitter(split_mode='FROM_END',
                                            num_items=num_items)
        train, test = splitter.get_train_test_indices(10)
        self.assertListEqual(train, [0, 1, 2, 3, 4, 5])
        self.assertListEqual(test, [6, 7, 8, 9])

    def test_get_indices_with_split_mode_from_end_with_fraction(self):
        frac = 0.2
        splitter = netutils.DatasetSplitter(split_mode='FROM_END',
                                            items_fraction=frac)
        train, test = splitter.get_train_test_indices(10)
        self.assertListEqual(train, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(test, [8, 9])

    # Currently no real way to test the 'RANDOM' mode, aside from checking
    # the generated train/test indices. We just check that the indices are
    # disjoint and the lengths of both index lists are as expected.
    def test_get_indices_with_split_mode_random_with_num(self):
        num_items = 4
        splitter = netutils.DatasetSplitter(split_mode='RANDOM',
                                            num_items=num_items)
        train, test = splitter.get_train_test_indices(10)
        self.assertTrue(set(train).isdisjoint(set(test)))
        self.assertEqual(len(train), 10 - num_items)
        self.assertEqual(len(test), num_items)

    def test_get_indices_with_split_mode_random_with_fraction(self):
        frac = 0.2
        splitter = netutils.DatasetSplitter(split_mode='RANDOM',
                                            items_fraction=frac)
        train, test = splitter.get_train_test_indices(10)
        self.assertTrue(set(train).isdisjoint(set(test)))
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)

    def test_num_overrides_fraction(self):
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            items_fraction=0.6,
                                            num_items=2)
        train, test = splitter.get_train_test_indices(10)
        self.assertListEqual(train, [2, 3, 4, 5, 6, 7, 8, 9])
        self.assertListEqual(test, [0, 1])

    def test_get_data_and_targets(self):
        #self.maxDiff = None
        dset = self.dset
        exp_res = {
            'test_data': dset.get_data_as_arraylike(slice(2)),
            'test_targets': dset.get_targets(slice(2)),
            'train_data': dset.get_data_as_arraylike(slice(2, None)),
            'train_targets': dset.get_targets(slice(2, None))}
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            items_fraction=0.6,
                                            num_items=2)
        res = splitter.get_data_and_targets(dset)
        self.assertDictEqual(res, exp_res)

    def test_get_data_and_targets_test_dset_overrides_num_and_fraction(self):
        dset = self.dset
        exp_res = {
            'test_data': dset.get_data_as_arraylike(),
            'test_targets': dset.get_targets(),
            'train_data': dset.get_data_as_arraylike(),
            'train_targets': dset.get_targets()}
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            items_fraction=0.3,
                                            num_items=2)
        res = splitter.get_data_and_targets(dset, test_dset=dset)
        self.assertDictEqual(res, exp_res)

    def test_invalid_split_mode_raises_error(self):
        self.assertRaises(ValueError, netutils.DatasetSplitter,
                          split_mode='TEST')

    def test_negative_fraction_raises_error(self):
        self.assertRaises(ValueError, netutils.DatasetSplitter,
                          split_mode='FROM_START', items_fraction=-0.1)

    def test_fraction_greater_than_one_raises_error(self):
        self.assertRaises(ValueError, netutils.DatasetSplitter,
                          split_mode='FROM_START', items_fraction=1.6)


if __name__ == '__main__':
    unittest.main()
