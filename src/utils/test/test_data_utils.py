import unittest

import numpy as np
import numpy.testing as nptest

import test.test_setups as testset
import utils.data_utils as ds

class TestDatasetUtilsFunctions(testset.DatasetItemsMixin, unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
