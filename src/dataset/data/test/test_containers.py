import unittest

import numpy as np

import dataset.data.constants as cons
import dataset.data.containers as utils


class TestContainerCreatorFunctions(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        f_w, f_h, n_f = 48, 64, 20
        packet_shape = (n_f, f_h, f_w)
        item_shapes = {
            'raw' : (n_f, f_h, f_w),
            'yx'  : (f_h, f_w),
            'gtux': (n_f, f_w),
            'gtuy': (n_f, f_h),
        }

        cls.n_packets = 2
        cls.packet_shape = packet_shape
        cls.item_shapes = item_shapes

    # test create data item containers (known number of items)
    # here we do not assert that the returned ndarray equals
    # another array, we just test if its shape is as expected

    def test_create_packet_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['raw'])
        result = utils.create_packet_container(self.packet_shape,
                                               num_items=self.n_packets)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_y_x_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['yx'])
        result = utils.create_y_x_projection_container(self.packet_shape,
                                                       num_items=self.n_packets)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_gtu_x_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['gtux'])
        result = utils.create_gtu_x_projection_container(self.packet_shape,
                                                         num_items=self.n_packets)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_gtu_y_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['gtuy'])
        result = utils.create_gtu_y_projection_container(self.packet_shape,
                                                         num_items=self.n_packets)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_data_holders(self):
        item_types = {k: True for k in cons.ALL_ITEM_TYPES}
        exp_shapes = {k: (self.n_packets, *self.item_shapes[k])
                      for k in cons.ALL_ITEM_TYPES}
        item_types['gtux'] = False
        del exp_shapes['gtux']

        holders = utils.create_data_containers(self.packet_shape, item_types,
                                               num_items=self.n_packets)
        holder_shapes = {k: (v.shape if v is not None else None)
                         for k, v in holders.items()}
        self.assertDictEqual(holder_shapes, exp_shapes)

    # test create data item holders (unknown number of items)

    def test_create_packet_holder_unknown_num_packets(self):
        result = utils.create_packet_container(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_y_x_projection_holder_unknown_num_packets(self):
        result = utils.create_y_x_projection_container(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_gtu_x_projection_holder_unknown_num_packets(self):
        result = utils.create_gtu_x_projection_container(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_gtu_y_projection_holder_unknown_num_packets(self):
        result = utils.create_gtu_y_projection_container(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_data_holders_unknown_num_packets(self):
        item_types = {k: True for k in cons.ALL_ITEM_TYPES}
        exp_holders = {k: [] for k in cons.ALL_ITEM_TYPES}
        item_types['yx'] = False
        del exp_holders['yx']

        holders = utils.create_data_containers(self.packet_shape, item_types)
        self.assertDictEqual(holders, exp_holders)


if __name__ == '__main__':
    unittest.main()
