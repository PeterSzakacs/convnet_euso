import unittest

import numpy as np
import numpy.testing as nptest

import dataset.constants as cons
import dataset.dataset_utils as ds
import test.test_setups as testset


class TestNumpyDataset(testset.DatasetItemsMixin, testset.DatasetTargetsMixin,
                       testset.DatasetMetadataMixin, unittest.TestCase):

    # helper methods (custom assert)

    def _assertDatasetData(self, data, exp_data, exp_item_types):
        # unfortunately, assertDictEqual does not work in this case
        self.assertSetEqual(set(data.keys()), set(exp_data.keys()),
                            "Returned item keys not equal")
        for k in exp_data.keys():
            nptest.assert_array_equal(data[k], exp_data[k])
            # code below is slightly faster, but non-standard
            # self.assertTrue(np.array_equal(data[k], exp_data[k]),
            #                 "item type '{}' not equal".format(k))

    def _assertDatasetTargets(self, targets, exp_targets):
        msg = "Targets not equal: expected {}:, actual {}:".format(
            exp_targets, targets)
        nptest.assert_array_equal(targets, exp_targets, msg)

    # NOTE: This assert is rather slow (~10-20ms), maybe the individual asserts
    # could be in their own tests
    def _assertDatasetItems(self, dset, exp_data, exp_targets, exp_meta,
                            exp_metafields, exp_num_data, exp_item_types):
        data = dset.get_data_as_dict()
        self._assertDatasetData(data, exp_data, exp_item_types)
        targets = dset.get_targets()
        self._assertDatasetTargets(targets, exp_targets)
        meta = dset.get_metadata()
        self.assertListEqual(meta, exp_meta, "Metadata not equal")
        self.assertSetEqual(dset.metadata_fields, exp_metafields)
        self.assertEqual(dset.num_data, exp_num_data, "Num data not equal")

    # test setup

    @classmethod
    def setUpClass(cls):
        super(TestNumpyDataset, cls).setUpClass()
        cls.name = 'test'

    # test dataset item addition

    def test_add_item(self):
        dset = ds.NumpyDataset(self.name, self.packet_shape,
                               item_types=self.item_types)
        packet = self.items['raw'][0]
        exp_data = {k: [v[0]] for k, v in self.items.items()}
        num_data = dset.num_data

        dset.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        self.assertEqual(dset.num_data, num_data + 1)

    def test_add_item_non_resizable_dataset(self):
        dset = ds.NumpyDataset(self.name, self.packet_shape,
                               item_types=self.item_types)
        dset.resizable = False
        packet = self.items['raw'][0]
        targ, meta = self.mock_targets[0], self.mock_meta[0]

        self.assertRaises(Exception, dset.add_data_item, packet, targ, meta)

    def test_add_item_wrong_packet_shape(self):
        dset = ds.NumpyDataset(self.name, self.packet_shape,
                               item_types=self.item_types)
        packet = np.ones((1, *self.packet_shape))
        targ, meta = self.mock_targets[0], self.mock_meta[0]

        self.assertRaises(ValueError, dset.add_data_item, packet, targ, meta)

    # test dataset item getting

    def test_get_data_as_dict(self):
        item_types = {'raw': True, 'yx': True, 'gtux': False, 'gtuy': False}
        exp_items = {k: [self.items[k][0]] for k, v in item_types.items() if v}
        packet, target = self.items['raw'][0], self.mock_targets[0]
        meta = self.mock_meta[0]

        dset = ds.NumpyDataset(self.name, self.packet_shape,
                               item_types=item_types)
        dset.add_data_item(packet, target, meta)
        items = dset.get_data_as_dict()
        self._assertDatasetData(items, exp_items, exp_items.keys())

    def test_get_data_as_arraylike(self):
        keys = ('raw', 'yx')
        item_types = {'raw': True, 'yx': True, 'gtux': False, 'gtuy': False}
        exp_items = ([self.items['raw'][0]], [self.items['yx'][0]])
        packet, target = self.items['raw'][0], self.mock_targets[0]
        meta = self.mock_meta[0]

        dset = ds.NumpyDataset(self.name, self.packet_shape,
                               item_types=item_types)
        dset.add_data_item(packet, target, meta)
        items = dset.get_data_as_arraylike()
        for idx in range(len(keys)):
            err_msg = "items of type '{}' are not equal".format(keys[idx])
            nptest.assert_array_equal(items[idx], exp_items[idx], err_msg)

    def test_get_targets(self):
        packet, target = self.items['raw'][0], self.mock_targets[0]
        meta = self.mock_meta[0]
        exp_targets = [self.mock_targets[0]]

        dset = ds.NumpyDataset(self.name, self.packet_shape)
        dset.add_data_item(packet, target, meta)
        targets = dset.get_targets()
        self._assertDatasetTargets(targets, exp_targets)

    def test_get_metadata(self):
        packet, target = self.items['raw'][0], self.mock_targets[0]
        meta = self.mock_meta[0]
        exp_metadata = [self.mock_meta[0]]

        dset = ds.NumpyDataset(self.name, self.packet_shape)
        dset.add_data_item(packet, target, meta)
        metadata = dset.get_metadata()
        msg = "Metadata not equal: expected {}:, actual {}:".format(
            exp_metadata, meta)
        self.assertListEqual(metadata, exp_metadata, msg)

    # test merging datasets

    def test_merge_with(self):
        dset1 = ds.NumpyDataset(self.name, self.packet_shape,
                                item_types=self.item_types)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape,
                                item_types=self.item_types)
        packet = self.items['raw'][0]
        dset1.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        dset2.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        num_data = dset1.num_data
        exp_data = { 'raw': [packet, packet],
                     'yx': [self.items['yx'][0], self.items['yx'][0]],
                     'gtux': [self.items['gtux'][0], self.items['gtux'][0]],
                     'gtuy': [self.items['gtuy'][0], self.items['gtuy'][0]] }
        exp_targets = [self.mock_targets[0], self.mock_targets[0]]
        exp_metadata = [self.mock_meta[0], self.mock_meta[0]]
        exp_metafields = self.metafields
        dset1.merge_with(dset2)
        self._assertDatasetItems(dset1, exp_data, exp_targets, exp_metadata,
                                 exp_metafields, num_data + dset2.num_data,
                                 cons.ALL_ITEM_TYPES)

    def test_merge_with_new_metafields(self):
        dset1 = ds.NumpyDataset(self.name, self.packet_shape)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape)
        packet = self.items['raw'][0]
        meta2 = self.mock_meta[0].copy()
        meta2['test'] = 'value'
        exp_metafields = self.metafields.union(meta2.keys())
        dset1.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        dset2.add_data_item(packet, self.mock_targets[0], meta2)
        dset1.merge_with(dset2)
        self.assertSetEqual(dset1.metadata_fields, exp_metafields)

    def test_merge_with_only_subset_of_items(self):
        dset1 = ds.NumpyDataset(self.name, self.packet_shape)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape)
        packet = self.items['raw'][0]
        meta2 = self.mock_meta[0].copy()
        meta2['test'] = 'value'
        meta3 = self.mock_meta[0].copy()
        meta3['test2'] = 'value'
        exp_metafields = self.metafields.union(meta2.keys())
        dset1.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        dset2.add_data_item(packet, self.mock_targets[0], meta2)
        dset2.add_data_item(packet, self.mock_targets[0], meta2)
        dset2.add_data_item(packet, self.mock_targets[0], meta3)
        # add items 0 and 1 from dset2 to dset1
        dset1.merge_with(dset2, slice(2))
        # metadata fields from item 2 of dset2 should not be added
        exp_metafields = self.metafields.union(meta2.keys())
        self.assertEqual(dset1.num_data, 3)
        self.assertSetEqual(dset1.metadata_fields, exp_metafields)

    def test_merge_with_not_resizable(self):
        dset1 = ds.NumpyDataset(self.name, self.packet_shape)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape)
        dset1.resizable = False
        packet = self.items['raw'][0]
        dset2.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        self.assertRaises(Exception, dset1.merge_with, dset2)

    def test_merge_with_incompatible_dataset(self):
        bad_packet_shape = (self.n_f + 1, self.f_h, self.f_h)
        dset1 = ds.NumpyDataset(self.name, bad_packet_shape)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape)
        packet = self.items['raw'][0]
        dset2.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        self.assertRaises(ValueError, dset1.merge_with, dset2)

    # test dataset compatibility checking

    def test_is_compatible_with(self):
        dset1 = ds.NumpyDataset(self.name, self.packet_shape)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape)
        self.assertTrue(dset1.is_compatible_with(dset2))

    def test_is_compatible_with_bad_packet_shape(self):
        bad_packet_shape = (self.n_f + 1, self.f_h, self.f_h)
        dset1 = ds.NumpyDataset(self.name, bad_packet_shape)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape)
        self.assertFalse(dset1.is_compatible_with(dset2))

    def test_is_compatible_with_bad_item_types(self):
        bad_item_types = self.item_types.copy()
        bad_item_types['raw'] = not bad_item_types['raw']
        dset1 = ds.NumpyDataset(self.name, self.packet_shape,
                                item_types=bad_item_types)
        dset2 = ds.NumpyDataset(self.name, self.packet_shape,
                                item_types=self.item_types)
        self.assertFalse(dset1.is_compatible_with(dset2))

    # test adding new metafield with default value

    def test_add_metafield(self):
        dset = ds.NumpyDataset(self.name, self.packet_shape)
        packet = self.items['raw'][0]
        exp_meta = self.mock_meta.copy()
        exp_meta[0] = exp_meta[0].copy()
        exp_meta[1] = exp_meta[1].copy()
        dset.add_data_item(packet, self.mock_targets[0], exp_meta[0])
        dset.add_data_item(packet, self.mock_targets[1], exp_meta[1])
        exp_meta = exp_meta.copy()
        exp_meta[0] = exp_meta[0].copy()
        exp_meta[1] = exp_meta[1].copy()
        exp_meta[0]['random_metafield'] = 'default'
        exp_meta[1]['random_metafield'] = 'default'
        exp_metafields = self.metafields.union(['random_metafield'])

        dset.add_metafield('random_metafield', default_value='default')
        self.assertListEqual(dset.get_metadata(), exp_meta)
        self.assertSetEqual(dset.metadata_fields, exp_metafields)

    # dtype functionality

    def test_implicit_dtype_conversion_when_adding_items(self):
        dset = ds.NumpyDataset(self.name, self.packet_shape, dtype='float16',
                               item_types=self.item_types)
        dset.add_data_item(self.items['raw'][0], self.mock_targets[0],
                           self.mock_meta[0])
        items_dict = dset.get_data_as_dict()
        for itype, is_present in dset.item_types.items():
            if is_present:
                self.assertEqual(items_dict[itype][0].dtype.name, 'float16')

    def test_dtype_casting(self):
        dset = ds.NumpyDataset(self.name, self.packet_shape,
                               item_types=self.item_types)
        dset.add_data_item(self.items['raw'][0], self.mock_targets[0],
                           self.mock_meta[0])
        dset.dtype = 'float16'
        items_dict = dset.get_data_as_dict()
        for itype, is_present in dset.item_types.items():
            if is_present:
                self.assertEqual(items_dict[itype][0].dtype.name, 'float16')
        self.assertEqual(dset.dtype, 'float16')

    # TODO: Possibly think of a way to test ds.shuffle_dataset,
    # though it has low priority


if __name__ == '__main__':
    unittest.main()
