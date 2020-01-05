import unittest
import unittest.mock as mock

import numpy as np
import numpy.testing as nptest

import dataset.tck.io_utils as io_utils
import utils.data_templates as templates


class TestPacketExtractor(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        # mock filenames
        cls.srcfile, cls.triggerfile = "srcfile", None
        # packet template for the functions
        ec_width, ec_height = 16, 32
        width, height, num_frames = 48, 64, 20
        cls.template = templates.PacketTemplate(ec_width, ec_height,
                                                width, height, num_frames)
        # the packets as used in the functions
        n_packets = 2
        cls.packets = np.ones((n_packets * num_frames, height, width),
                               dtype=np.uint8)
        cls.packets[0, 0, 0] = 10
        cls.packets[num_frames, 0, 0] = 2
        cls.expected_packets_shape = (n_packets, num_frames, height, width)
        cls.expected_packets = cls.packets.reshape(cls.expected_packets_shape)
        # the extractor object
        cls.extractor = io_utils.PacketExtractor(packet_template=cls.template)

    # test methods

    @mock.patch('numpy.load')
    def test_extract_from_npyfile(self, m_np_load):
        m_np_load.return_value = self.packets
        extracted_packets = self.extractor.extract_packets_from_npyfile(
            self.srcfile, self.triggerfile)
        m_np_load.assert_called_with(self.srcfile)
        nptest.assert_array_equal(extracted_packets, self.expected_packets)

    @mock.patch('dataset.tck.io_utils.reading')
    def test_extract_from_rootfile(self, m_reading):
        it = NpyIterator(self.packets)
        m_reading.AcqL1EventReader.return_value = it
        extracted_packets = self.extractor.extract_packets_from_rootfile(
            self.srcfile, self.triggerfile)
        m_reading.AcqL1EventReader.assert_called_with(
            self.srcfile, self.triggerfile)
        nptest.assert_array_equal(extracted_packets, self.expected_packets)


# mock class for utils.event_reading.GtuPdmDataIterator
# and utils.event_reading.AcqL1EventReader
class NpyIterator:

    # mock class for utils.event_reading.GtuPdmDataObject
    class Frame:

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
            return NpyIterator.Frame(
                self.index, self.packets[self.index].reshape(self.new_shape)
            )
        else:
            raise StopIteration()


if __name__ == '__main__':
    unittest.main()
