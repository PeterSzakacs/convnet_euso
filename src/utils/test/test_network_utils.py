import unittest

import test.test_setups as setups
import utils.network_utils as netutils


class TestDatasetSplitter(setups.DatasetMixin, unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        super(TestDatasetSplitter, cls).setUpClass(num_items=10, item_types={
            'yx': True, 'gtux': False, 'gtuy': True, 'raw': False})
        dset = cls.dset
        fraction, num = 0.3, 3
        cls.first_items = dset.get_data_as_arraylike(slice(num))
        cls.last_items = dset.get_data_as_arraylike(slice(num, 10))
        cls.first_targs = dset.get_targets(slice(num))
        cls.last_targs = dset.get_targets(slice(num, 10))
        cls.fraction, cls.num = fraction, num

    # test methods

    def test_split_mode_from_start_with_num(self):
        num_items, dset = self.num, self.dset
        exp_res = {
            'test_data': self.first_items,
            'test_targets': self.first_targs,
            'train_data': self.last_items,
            'train_targets': self.last_targs}
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            num_items=num_items)
        res = splitter.get_data_and_targets(dset)
        self.assertDictEqual(res, exp_res)

    def test_split_mode_from_start_with_fraction(self):
        frac, dset = self.fraction, self.dset
        exp_res = {
            'test_data': self.first_items,
            'test_targets': self.first_targs,
            'train_data': self.last_items,
            'train_targets': self.last_targs}
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            items_fraction=frac)
        res = splitter.get_data_and_targets(dset)
        self.assertDictEqual(res, exp_res)

    def test_split_mode_from_end_with_num(self):
        num_items, dset = self.num, self.dset
        exp_res = {
            'test_data': self.last_items,
            'test_targets': self.last_targs,
            'train_data': self.first_items,
            'train_targets': self.first_targs}
        splitter = netutils.DatasetSplitter(split_mode='FROM_END',
                                            num_items=num_items)
        res = splitter.get_data_and_targets(dset)
        self.assertDictEqual(res, exp_res)

    def test_split_mode_from_end_with_fraction(self):
        frac, dset = self.fraction, self.dset
        exp_res = {
            'test_data': self.last_items,
            'test_targets': self.last_targs,
            'train_data': self.first_items,
            'train_targets': self.first_targs}
        splitter = netutils.DatasetSplitter(split_mode='FROM_END',
                                            items_fraction=frac)
        res = splitter.get_data_and_targets(dset)
        self.assertDictEqual(res, exp_res)

    # Currently no real way to test the 'RANDOM' mode, aside from checking
    # the internally generated train/test indexes, which requires exposing
    # that functonality. For now its just excluded from testing.
    def test_dset_split_mode_random_with_num(self):
        num_items, dset = self.num, self.dset
        keys = ('test_data', 'test_targets', 'train_data', 'train_targets', )
        splitter = netutils.DatasetSplitter(split_mode='RANDOM',
                                            num_items=num_items)
        res = splitter.get_data_and_targets(dset)
        self.assertSetEqual(set(res.keys()), set(keys))
        self.assertEqual(len(res['train_data'][0]), dset.num_data - num_items)
        self.assertEqual(len(res['train_targets']), dset.num_data - num_items)
        self.assertEqual(len(res['test_data'][0]), num_items)
        self.assertEqual(len(res['test_targets']), num_items)

    def test_dset_split_mode_random_with_fraction(self):
        frac, dset = self.fraction, self.dset
        num_items = round(dset.num_data * frac)
        keys = ('test_data', 'test_targets', 'train_data', 'train_targets', )
        splitter = netutils.DatasetSplitter(split_mode='RANDOM',
                                            items_fraction=frac)
        res = splitter.get_data_and_targets(dset)
        self.assertSetEqual(set(res.keys()), set(keys))
        self.assertEqual(len(res['train_data'][0]), dset.num_data - num_items)
        self.assertEqual(len(res['train_targets']), dset.num_data - num_items)
        self.assertEqual(len(res['test_data'][0]), num_items)
        self.assertEqual(len(res['test_targets']), num_items)

    def test_num_overrides_faction(self):
        num_items, frac, dset = self.num, self.fraction, self.dset
        exp_res = {
            'test_data': self.first_items,
            'test_targets': self.first_targs,
            'train_data': self.last_items,
            'train_targets': self.last_targs}
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            items_fraction=(frac + 0.3),
                                            num_items=num_items)
        res = splitter.get_data_and_targets(dset)
        self.assertDictEqual(res, exp_res)

    def test_test_dset_overrides_all_else(self):
        num_items, frac, dset = self.num, self.fraction, self.dset
        exp_res = {
            'test_data': dset.get_data_as_arraylike(),
            'test_targets': dset.get_targets(),
            'train_data': dset.get_data_as_arraylike(),
            'train_targets': dset.get_targets()}
        splitter = netutils.DatasetSplitter(split_mode='FROM_START',
                                            items_fraction=frac,
                                            num_items=num_items)
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
