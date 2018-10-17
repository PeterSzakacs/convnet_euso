import unittest
import unittest.mock as mock

import numpy as np

import utils.dataset_utils as ds

@mock.patch('utils.dataset_utils.np')
class TestDatasetUtilsFunctions(unittest.TestCase):

    def setUp(self):
        self.f_w, self.f_h, self.n_f = 48, 64, 20
        self.packet_shape = (self.n_f, self.f_h, self.f_w)
        self.item_shapes = {
            'raw' : (self.n_f, self.f_h, self.f_w),
            'yx'  : (self.f_h, self.f_w),
            'gtux': (self.n_f, self.f_w),
            'gtuy': (self.n_f, self.f_h),
        }

    def test_holder_creation(self, mock_np):
        n_packets = 10
        holder_shapes = {k: (n_packets, *self.item_shapes[k]) 
                         for k in ds.ALL_ITEM_TYPES}
        packet_shape = self.packet_shape

        # First test individual holder creation functions
        ds.create_packet_holder(packet_shape, num_items=n_packets)
        mock_np.empty.assert_called_with(holder_shapes['raw'], dtype=np.uint8)
        ds.create_y_x_projection_holder(packet_shape, num_items=n_packets)
        mock_np.empty.assert_called_with(holder_shapes['yx'], dtype=np.uint8)
        ds.create_gtu_x_projection_holder(packet_shape, num_items=n_packets)
        mock_np.empty.assert_called_with(holder_shapes['gtux'], dtype=np.uint8)
        ds.create_gtu_y_projection_holder(packet_shape, num_items=n_packets)
        mock_np.empty.assert_called_with(holder_shapes['gtuy'], dtype=np.uint8)

        # Now test function which creates multiple data holders and returns 
        # them in a dict (based on which item type flags are set to True)
        side_effect = lambda shape, dtype: shape
        mock_np.empty.side_effect = side_effect
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in ds.ALL_ITEM_TYPES:
            holders = ds.create_data_holders(packet_shape, num_items=n_packets,
                                             dtype='a', item_types=item_types)
            self.assertDictEqual(holders, holder_shapes)
            holder_shapes[item_type] = None
            item_types[item_type] = False

    def test_packet_conversion(self, mock_np):
        packet = 'testtesttesttest'
        start, end = 3, 10
        sliced_packet = packet[start:end]
        num_gtu = end - start
        axis_to_item_keys = ('yx', 'gtux', 'gtuy')
        items = {
            'raw' : sliced_packet,
            'yx'  : (self.f_h, self.f_w),
            'gtux': (num_gtu, self.f_w),
            'gtuy': (num_gtu, self.f_h),
        }

        # First test individual packet conversion functions
        result = ds.create_subpacket(packet, start_idx=start, end_idx=end)
        self.assertEqual(result, sliced_packet)
        ds.create_y_x_projection(packet, start_idx=start, end_idx=end)
        mock_np.max.assert_called_with(sliced_packet, axis=0)
        ds.create_gtu_x_projection(packet, start_idx=start, end_idx=end)
        mock_np.max.assert_called_with(sliced_packet, axis=1)
        ds.create_gtu_y_projection(packet, start_idx=start, end_idx=end)
        mock_np.max.assert_called_with(sliced_packet, axis=2)

        # Now test function that converts a single packet into multiple items 
        # and returns them in a dict (based on which item type flags are set 
        # to True)
        side_effect = lambda packet, axis: items[axis_to_item_keys[axis]]
        mock_np.max.side_effect = side_effect
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in ds.ALL_ITEM_TYPES:
            ret_items = ds.convert_packet(packet, start_idx=start, end_idx=end,
                                          item_types=item_types)
            self.assertDictEqual(ret_items, items)
            items[item_type] = None
            item_types[item_type] = False

    def test_item_shape_getters(self, mock_np):
        packet_shape, item_shapes = self.packet_shape, self.item_shapes

        # First test individual item shape getters
        result = ds.get_y_x_projection_shape(packet_shape)
        self.assertTupleEqual(result, item_shapes['yx'])
        result = ds.get_gtu_x_projection_shape(packet_shape)
        self.assertTupleEqual(result, item_shapes['gtux'])
        result = ds.get_gtu_y_projection_shape(packet_shape)
        self.assertTupleEqual(result, item_shapes['gtuy'])

        # Now test function which gets item shapes based on which item types
        # are set to True
        item_shapes = item_shapes.copy()
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        for item_type in ds.ALL_ITEM_TYPES:
            shapes = ds.get_data_item_shapes(packet_shape, item_types)
            self.assertDictEqual(shapes, item_shapes)
            item_shapes[item_type] = None
            item_types[item_type] = False

    # def test_store_dataset(self, mock_np):
    #     data = ('1', '2', '3', '4', '5', '6')
    #     targets = 'targets'
    #     dataset_filenames = ('a', 'b', 'c', 'd', 'e', 'f')
    #     targets_filename = 'targetsfile'
    #     data_len = len(data)
    #     ds.save_dataset(data, targets, dataset_filenames, targets_filename)
    #     self.assertEqual(mock_np.save.call_count, data_len + 1)
    #     for idx in range(data_len):
    #         call = mock_np.save.call_args_list[idx]
    #         args = call[0]
    #         self.assertTupleEqual(args, (dataset_filenames[idx], data[idx]))
    #     call = mock_np.save.call_args_list[data_len]
    #     args = call[0]
    #     self.assertTupleEqual(args, (targets_filename, targets))

    # TODO: Possibly think of a way to test ds.shuffle_dataset,
    # though it has low priority


if __name__ == '__main__':
    unittest.main()
