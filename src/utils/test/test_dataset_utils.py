import unittest
import unittest.mock as mock

import numpy as np
import numpy.testing as nptest

import utils.dataset_utils as ds
import utils.metadata_utils as meta

class TestDatasetUtilsBase(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        cls.f_w, cls.f_h, cls.n_f = 48, 64, 20
        cls.packet_shape = (cls.n_f, cls.f_h, cls.f_w)
        cls.n_packets = 2
        cls.item_shapes = {
            'raw' : (cls.n_f, cls.f_h, cls.f_w),
            'yx'  : (cls.f_h, cls.f_w),
            'gtux': (cls.n_f, cls.f_w),
            'gtuy': (cls.n_f, cls.f_h),
        }
        cls.items = {
            'raw' : np.ones((cls.n_packets, *cls.item_shapes['raw'])),
            'yx'  : np.ones((cls.n_packets, *cls.item_shapes['yx'])),
            'gtux': np.ones((cls.n_packets, *cls.item_shapes['gtux'])),
            'gtuy': np.ones((cls.n_packets, *cls.item_shapes['gtuy']))}
        cls.item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        # make some randomness in the first packet and derived items
        packet = cls.items['raw'][0]
        packet[0, 0, 0], packet[1] = 3, 4
        yx = cls.items['yx'][0]
        yx.fill(4)
        gtux, gtuy = cls.items['gtux'][0], cls.items['gtuy'][0]
        gtux[0, 0], gtuy[0, 0], gtux[1], gtuy[1] = 3, 3, 4, 4


class TestDatasetUtilsFunctions(TestDatasetUtilsBase):

    # test setup

    @classmethod
    def setUpClass(cls):
        super(TestDatasetUtilsFunctions, cls).setUpClass()
        cls.packet = cls.items['raw'][0]
        cls.start, cls.end = 0, 10

    # test create data item holders

    def test_create_packet_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['raw'])
        result = ds.create_packet_holder(self.packet_shape,
                                         num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_y_x_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['yx'])
        result = ds.create_y_x_projection_holder(self.packet_shape,
                                                 num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_gtu_x_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['gtux'])
        result = ds.create_gtu_x_projection_holder(self.packet_shape,
                                                   num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_gtu_y_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['gtuy'])
        result = ds.create_gtu_y_projection_holder(self.packet_shape,
                                                   num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_data_holders(self):
        exp_shapes = {k: (self.n_packets, *self.item_shapes[k])
                         for k in ds.ALL_ITEM_TYPES}
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in ds.ALL_ITEM_TYPES:
            holders = ds.create_data_holders(self.packet_shape, item_types,
                                             num_items=self.n_packets)
            holder_shapes = {k: (v.shape if v is not None else None)
                             for k, v in holders.items()}
            self.assertDictEqual(holder_shapes, exp_shapes)
            exp_shapes[item_type] = None
            item_types[item_type] = False

    def test_create_packet_holder_unknown_num_packets(self):
        result = ds.create_packet_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_y_x_projection_holder_unknown_num_packets(self):
        result = ds.create_y_x_projection_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_gtu_x_projection_holder_unknown_num_packets(self):
        result = ds.create_gtu_x_projection_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_gtu_y_projection_holder_unknown_num_packets(self):
        result = ds.create_gtu_y_projection_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_data_holders(self):
        exp_holders = {k: [] for k in ds.ALL_ITEM_TYPES}
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in ds.ALL_ITEM_TYPES:
            holders = ds.create_data_holders(self.packet_shape, item_types)
            self.assertDictEqual(holders, exp_holders)
            exp_holders[item_type] = None
            item_types[item_type] = False

    # test create packet projections

    def test_create_subpacket(self):
        expected_result = self.items['raw'][0][self.start:self.end]
        result = ds.create_subpacket(self.packet, start_idx=self.start,
                                     end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_create_y_x_projection(self):
        expected_result = self.items['yx'][0]
        result = ds.create_y_x_projection(self.packet, start_idx=self.start,
                                          end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_create_gtu_x_projection(self):
        expected_result = self.items['gtux'][0][self.start:self.end]
        result = ds.create_gtu_x_projection(self.packet, start_idx=self.start,
                                            end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_create_gtu_y_projection(self):
        expected_result = self.items['gtuy'][0][self.start:self.end]
        result = ds.create_gtu_y_projection(self.packet, start_idx=self.start,
                                            end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_convert_packet(self):
        exp_items = {'raw': self.items['raw'][0][self.start:self.end],
            'gtux': self.items['gtux'][0][self.start:self.end],
            'gtuy': self.items['gtuy'][0][self.start:self.end],
            'yx'  : self.items['yx'][0]}
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        exp_types = {k: True for k in ds.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in ds.ALL_ITEM_TYPES:
            items = ds.convert_packet(self.packet, item_types,
                                      start_idx=self.start, end_idx=self.end)
            all_equal = {k: (not item_types[k] if v is None
                             else np.array_equal(v, exp_items[k]))
                             for k, v in items.items()}
            self.assertDictEqual(all_equal, exp_types)
            exp_items[item_type] = None
            item_types[item_type] = False

    # test get item shapes

    def test_get_y_x_projection_shape(self):
        result = ds.get_y_x_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['yx'])

    def test_get_gtu_x_projection_shape(self):
        result = ds.get_gtu_x_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['gtux'])

    def test_get_gtu_y_projection_shape(self):
        result = ds.get_gtu_y_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['gtuy'])

    def test_get_item_shape_gettters(self):
        # Test function which gets item shapes based on which item types are
        # set to True
        item_shapes = self.item_shapes.copy()
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        for item_type in ds.ALL_ITEM_TYPES:
            shapes = ds.get_data_item_shapes(self.packet_shape, item_types)
            self.assertDictEqual(shapes, item_shapes)
            item_shapes[item_type] = None
            item_types[item_type] = False


class TestNumpyDataset(TestDatasetUtilsBase):

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
        cls.mock_targets = np.zeros((cls.n_packets, 2))
        meta_dict = {k: None for k in meta.CLASS_METADATA}
        cls.mock_meta = [meta_dict] * cls.n_packets

    # test dataset item addition

    def test_add_item(self):
        dset = ds.numpy_dataset(self.name, self.packet_shape,
                                item_types=self.item_types)
        packet = self.items['raw'][0]
        exp_data = {k: [v[0]] for k, v in self.items.items()}
        num_data = dset.num_data

        dset.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        self.assertEqual(dset.num_data, num_data + 1)

    def test_add_item_non_resizable_dataset(self):
        dset = ds.numpy_dataset(self.name, self.packet_shape,
                                item_types=self.item_types)
        dset.resizable = False
        packet = self.items['raw'][0]
        targ, meta = self.mock_targets[0], self.mock_meta[0]

        self.assertRaises(Exception, dset.add_data_item, packet, targ, meta)

    def test_add_item_wrong_packet_shape(self):
        dset = ds.numpy_dataset(self.name, self.packet_shape,
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

        dset = ds.numpy_dataset(self.name, self.packet_shape, item_types)
        dset.add_data_item(packet, target, meta)
        items = dset.get_data_as_dict()
        self._assertDatasetData(items, exp_items, exp_items.keys())

    def test_get_data_as_arraylike(self):
        keys = ('raw', 'yx')
        item_types = {'raw': True, 'yx': True, 'gtux': False, 'gtuy': False}
        exp_items = ([self.items['raw'][0]], [self.items['yx'][0]])
        packet, target = self.items['raw'][0], self.mock_targets[0]
        meta = self.mock_meta[0]

        dset = ds.numpy_dataset(self.name, self.packet_shape, item_types)
        dset.add_data_item(packet, target, meta)
        items = dset.get_data_as_arraylike()
        for idx in range(len(keys)):
            err_msg = "items of type '{}' are not equal".format(keys[idx])
            nptest.assert_array_equal(items[idx], exp_items[idx], err_msg)

    def test_get_targets(self):
        packet, target = self.items['raw'][0], self.mock_targets[0]
        meta = self.mock_meta[0]
        exp_targets = [self.mock_targets[0]]

        dset = ds.numpy_dataset(self.name, self.packet_shape)
        dset.add_data_item(packet, target, meta)
        targets = dset.get_targets()
        self._assertDatasetTargets(targets, exp_targets)

    def test_get_metadata(self):
        packet, target = self.items['raw'][0], self.mock_targets[0]
        meta = self.mock_meta[0]
        exp_metadata = [self.mock_meta[0]]

        dset = ds.numpy_dataset(self.name, self.packet_shape)
        dset.add_data_item(packet, target, meta)
        metadata = dset.get_metadata()
        msg = "Metadata not equal: expected {}:, actual {}:".format(
            exp_metadata, meta)
        self.assertListEqual(metadata, exp_metadata, msg)

    # test merging datasets

    def test_merge_with(self):
        dset1 = ds.numpy_dataset(self.name, self.packet_shape, self.item_types)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape, self.item_types)
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
        exp_metafields = set(meta.CLASS_METADATA)
        dset1.merge_with(dset2)
        self._assertDatasetItems(dset1, exp_data, exp_targets, exp_metadata,
                                 exp_metafields, num_data + dset2.num_data,
                                 ds.ALL_ITEM_TYPES)

    def test_merge_with_new_metafields(self):
        dset1 = ds.numpy_dataset(self.name, self.packet_shape)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape)
        packet = self.items['raw'][0]
        meta2 = self.mock_meta[0].copy()
        meta2['test'] = 'value'
        exp_metafields = set(meta.CLASS_METADATA).union(meta2.keys())
        dset1.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        dset2.add_data_item(packet, self.mock_targets[0], meta2)
        dset1.merge_with(dset2)
        self.assertSetEqual(dset1.metadata_fields, exp_metafields)

    def test_merge_with_only_subset_of_items(self):
        dset1 = ds.numpy_dataset(self.name, self.packet_shape)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape)
        packet = self.items['raw'][0]
        meta2 = self.mock_meta[0].copy()
        meta2['test'] = 'value'
        meta3 = self.mock_meta[0].copy()
        meta3['test2'] = 'value'
        exp_metafields = set(meta.CLASS_METADATA).union(meta2.keys())
        dset1.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        dset2.add_data_item(packet, self.mock_targets[0], meta2)
        dset2.add_data_item(packet, self.mock_targets[0], meta2)
        dset2.add_data_item(packet, self.mock_targets[0], meta3)
        # add items 0 and 1 from dset2 to dset1
        dset1.merge_with(dset2, slice(2))
        # metadata fields from item 2 of dset2 should not be added
        exp_metafields = set(meta.CLASS_METADATA).union(meta2.keys())
        self.assertEqual(dset1.num_data, 3)
        self.assertSetEqual(dset1.metadata_fields, exp_metafields)

    def test_merge_with_not_resizable(self):
        dset1 = ds.numpy_dataset(self.name, self.packet_shape)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape)
        dset1.resizable = False
        packet = self.items['raw'][0]
        dset2.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        self.assertRaises(Exception, dset1.merge_with, dset2)

    def test_merge_with_incompatible_dataset(self):
        bad_packet_shape = (self.n_f + 1, self.f_h, self.f_h)
        dset1 = ds.numpy_dataset(self.name, bad_packet_shape)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape)
        packet = self.items['raw'][0]
        dset2.add_data_item(packet, self.mock_targets[0], self.mock_meta[0])
        self.assertRaises(ValueError, dset1.merge_with, dset2)

    # test dataset compatibility checking

    def test_is_compatible_with(self):
        dset1 = ds.numpy_dataset(self.name, self.packet_shape)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape)
        self.assertTrue(dset1.is_compatible_with(dset2))

    def test_is_compatible_with_bad_packet_shape(self):
        bad_packet_shape = (self.n_f + 1, self.f_h, self.f_h)
        dset1 = ds.numpy_dataset(self.name, bad_packet_shape)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape)
        self.assertFalse(dset1.is_compatible_with(dset2))

    def test_is_compatible_with_bad_item_types(self):
        bad_item_types = self.item_types.copy()
        bad_item_types['raw'] = not bad_item_types['raw']
        dset1 = ds.numpy_dataset(self.name, self.packet_shape, bad_item_types)
        dset2 = ds.numpy_dataset(self.name, self.packet_shape, self.item_types)
        self.assertFalse(dset1.is_compatible_with(dset2))

    # TODO: Possibly think of a way to test ds.shuffle_dataset,
    # though it has low priority


if __name__ == '__main__':
    unittest.main()
