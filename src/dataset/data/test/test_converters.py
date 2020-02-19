import numpy as np
import numpy.testing as nptest

import dataset.data.constants as cons
import dataset.data.converters as conv
import dataset.data.test.utils as test_utils


class TestPacketConverterFunctions(test_utils.DataItemsDictUtilsMixin):

    # test setup

    @classmethod
    def setUpClass(cls):
        packet = np.ones((100, 10, 20), dtype=np.uint8)
        start, end = 1, 10
        items = {
            "raw": packet[start:end],
            "yx": np.ones((10, 20)),
            "gtux": np.ones((9, 20)),
            "gtuy": np.ones((9, 10)),
        }
        cls.packet = packet
        cls.items = items
        cls.start, cls.end = start, end

    # test packet conversion functions

    def test_create_subpacket(self):
        expected_result = self.items['raw']
        result = conv.create_subpacket(self.packet,
                                       start_idx=self.start,
                                       end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, expected_result.dtype)

    def test_create_y_x_projection_max(self):
        expected_result = self.items['yx']
        result = conv.create_y_x_projection_max(self.packet,
                                                start_idx=self.start,
                                                end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, expected_result.dtype)

    def test_create_gtu_x_projection_max(self):
        expected_result = self.items['gtux']
        result = conv.create_gtu_x_projection_max(self.packet,
                                                  start_idx=self.start,
                                                  end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, expected_result.dtype)

    def test_create_gtu_y_projection_max(self):
        expected_result = self.items['gtuy']
        result = conv.create_gtu_y_projection_max(self.packet,
                                                  start_idx=self.start,
                                                  end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, expected_result.dtype)

    def test_convert_packet(self):
        item_types = {k: True for k in cons.ALL_ITEM_TYPES}
        exp_items = self.items.copy()
        item_types['yx'] = False
        del exp_items['yx']

        items = conv.convert_packet(self.packet, item_types,
                                    start_idx=self.start, end_idx=self.end)
        self.assertItemsDictEqual(items, exp_items)

    def test_create_subpacket_new_dtype(self):
        expected_result = self.items['raw']
        result = conv.create_subpacket(self.packet, dtype='uint8',
                                       start_idx=self.start,
                                       end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, np.uint8)

    def test_create_y_x_projection_max_new_dtype(self):
        expected_result = self.items['yx']
        result = conv.create_y_x_projection_max(self.packet, dtype='uint8',
                                                start_idx=self.start,
                                                end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, np.uint8)

    def test_create_gtu_x_projection_max_new_dtype(self):
        expected_result = self.items['gtux']
        result = conv.create_gtu_x_projection_max(self.packet, dtype='uint8',
                                                  start_idx=self.start,
                                                  end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, np.uint8)

    def test_create_gtu_y_projection_max_new_dtype(self):
        expected_result = self.items['gtuy']
        result = conv.create_gtu_y_projection_max(self.packet, dtype='uint8',
                                                  start_idx=self.start,
                                                  end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)
        self.assertEqual(result.dtype, np.uint8)

    def test_convert_packet_new_dtype(self):
        item_types = {k: True for k in cons.ALL_ITEM_TYPES}
        item_types['gtux'] = False

        exp_items = self.items.copy()
        exp_items['raw'] = exp_items['raw'].astype(np.uint8)
        exp_items['yx'] = exp_items['yx'].astype(np.uint8)
        exp_items['gtuy'] = exp_items['gtuy'].astype(np.uint8)
        del exp_items['gtux']

        items = conv.convert_packet(self.packet, item_types, dtype='uint8',
                                    start_idx=self.start, end_idx=self.end)
        self.assertItemsDictEqual(items, exp_items)
