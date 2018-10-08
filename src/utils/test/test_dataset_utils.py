import unittest
import unittest.mock as mock

import utils.dataset_utils as ds

@mock.patch('utils.dataset_utils.np')
class TestDatasetUtils(unittest.TestCase):

    def setUp(self):
        self.outputs = ('raw', 'yx', 'gtux', 'gtuy')
        self.f_w, self.f_h, self.n_f = 48, 64, 20
        self.packet_shape = (self.n_f, self.f_h, self.f_w)

    def test_holder_creation(self, mock_np):
        n_packets = 10
        holder_shapes = [(n_packets, self.n_f, self.f_h, self.f_w),
                         (n_packets, self.f_h, self.f_w),
                         (n_packets, self.n_f, self.f_w),
                         (n_packets, self.n_f, self.f_h)]
        flags = [True] * len(self.outputs)
        helper = ds.numpy_dataset_helper(*flags)
        packet_shape = self.packet_shape

        # First test individual holder creation methods
        helper.create_subpacket_holder(n_packets, packet_shape)
        mock_np.empty.assert_called_with(holder_shapes[0])
        helper.create_y_x_projection_holder(n_packets, packet_shape)
        mock_np.empty.assert_called_with(holder_shapes[1])
        helper.create_gtu_x_projection_holder(n_packets, packet_shape)
        mock_np.empty.assert_called_with(holder_shapes[2])
        helper.create_gtu_y_projection_holder(n_packets, packet_shape)
        mock_np.empty.assert_called_with(holder_shapes[3])

        # Now test method which creates multiple data holders and returns them
        # in a tuple (based on which output flags are set to True)
        side_effect = lambda shape: shape
        mock_np.empty.side_effect = side_effect
        ## gradually turn off all outputs except 'gtuy'
        for idx in range(len(self.outputs)):
            helper.output_raw = flags[0]
            helper.output_yx = flags[1]
            helper.output_gtux = flags[2]
            helper.output_gtuy = flags[3]
            holders = helper.create_converted_packets_holders(n_packets,
                                                              packet_shape)
            expected_holders = tuple(holder_shapes[idx]
                                     for idx in range(len(holder_shapes))
                                     if flags[idx] == True)
            self.assertTupleEqual(holders, expected_holders)
            flags[idx] = False

    def test_packet_conversion(self, mock_np):
        packet = 'testtesttesttest'
        start, end = 3, 10
        sliced_packet = packet[start:end]
        num_gtu = end - start
        output_items = [sliced_packet, (self.f_h, self.f_w),
                         (num_gtu, self.f_w), (num_gtu, self.f_h)]
        flags = [True] * len(self.outputs)
        helper = ds.numpy_dataset_helper(*flags)

        # First test individual packet conversion methods
        result = helper.create_subpacket(packet, start_idx=start, end_idx=end)
        self.assertEqual(result, sliced_packet)
        helper.create_y_x_projection(packet, start_idx=start, end_idx=end)
        mock_np.max.assert_called_with(sliced_packet, axis=0)
        helper.create_gtu_x_projection(packet, start_idx=start, end_idx=end)
        mock_np.max.assert_called_with(sliced_packet, axis=1)
        helper.create_gtu_y_projection(packet, start_idx=start, end_idx=end)
        mock_np.max.assert_called_with(sliced_packet, axis=2)

        # Now test method that converts a single packet into multiple items and
        # returns them in a tuple (based on which output flags are set to True)
        side_effect = lambda packet, axis: output_items[axis+1]
        mock_np.max.side_effect = side_effect
        ## gradually turn off all outputs except 'gtuy'
        for idx in range(len(self.outputs)):
            helper.output_raw = flags[0]
            helper.output_yx = flags[1]
            helper.output_gtux = flags[2]
            helper.output_gtuy = flags[3]
            items = helper.convert_packet(packet, start_idx=start, end_idx=end)
            expected_items = tuple(output_items[idx]
                                   for idx in range(len(output_items))
                                   if flags[idx] == True)
            self.assertTupleEqual(items, expected_items)
            flags[idx] = False

    def test_store_dataset(self, mock_np):
        data = ('1', '2', '3', '4', '5', '6')
        targets = 'targets'
        dataset_filenames = ('a', 'b', 'c', 'd', 'e', 'f')
        targets_filename = 'targetsfile'
        data_len = len(data)
        ds.save_dataset(data, targets, dataset_filenames, targets_filename)
        self.assertEqual(mock_np.save.call_count, data_len + 1)
        for idx in range(data_len):
            call = mock_np.save.call_args_list[idx]
            args = call[0]
            self.assertTupleEqual(args, (dataset_filenames[idx], data[idx]))
        call = mock_np.save.call_args_list[data_len]
        args = call[0]
        self.assertTupleEqual(args, (targets_filename, targets))

    # TODO: Possibly think of a way to test ds.shuffle_dataset,
    # though it has low priority


if __name__ == '__main__':
    unittest.main()