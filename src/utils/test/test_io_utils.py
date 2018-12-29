import unittest
import unittest.mock as mock

import numpy as np

import utils.data_templates as templates
import utils.io_utils as io_utils

# mock class for utils.event_reading.GtuPdmDataIterator and .AcqL1EventReader
class npyIterator():

    # mock class for utils.event_reading.GtuPdmDataObject
    class frame():

        def __init__(self, gtu, photon_count_data):
            self.gtu = gtu
            self.photon_count_data = photon_count_data

    def __init__(self, packets_list):
        self.packets = packets_list
        frame_h, frame_w = packets_list.shape[1:]
        self.new_shape = (1, 1, frame_h, frame_w)
        self.tevent_entries = len(packets_list)
        self.index = -1

    def iter_gtu_pdm_data(self):
        self.index = -1
        return self

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index < self.tevent_entries:
            return npyIterator.frame(self.index, self.packets[self.index]
                                     .reshape(self.new_shape))
        else:
            raise StopIteration()

class testPacketExtraction(unittest.TestCase):

    # test setup

    def setUp(self):
        # mock filenames
        self.srcfile, self.triggerfile = "srcfile", None
        # packet template for the functions
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        self.template = templates.packet_template(EC_width, EC_height,
                                                  width, height, num_frames)
        # the packets as used in the functions
        n_packets = 2
        self.packets = np.ones((n_packets * num_frames, height, width),
                               dtype=np.uint8)
        self.packets[0, 0, 0] = 10
        self.packets[num_frames, 0, 0] = 2
        self.expected_packets_shape = (n_packets, num_frames, height, width)
        # the extractor object
        self.extractor = io_utils.packet_extractor(
            packet_template=self.template)

    # test methods

    @mock.patch('utils.io_utils.np')
    def test_extract_from_npyfile(self, mock_np):
        mock_np.load.return_value = self.packets
        extracted_packets = self.extractor.extract_packets_from_npyfile(
            self.srcfile, self.triggerfile)
        mock_np.load.assert_called_with(self.srcfile)
        self.assertTrue(np.array_equal(
            self.packets.reshape(self.expected_packets_shape),
            extracted_packets))

    @mock.patch('utils.io_utils.reading')
    def test_extract_from_rootfile(self, mock_reading):
        # after careful consideration, mocking np within this method makes no
        # sense
        it = npyIterator(self.packets)
        mock_reading.AcqL1EventReader.return_value = it
        extracted_packets = self.extractor.extract_packets_from_rootfile(
            self.srcfile, self.triggerfile)
        mock_reading.AcqL1EventReader.assert_called_with(self.srcfile,
                                                         self.triggerfile)
        self.assertTrue(np.array_equal(
            self.packets.reshape(self.expected_packets_shape),
            extracted_packets))


if __name__ == '__main__':
    unittest.main()

