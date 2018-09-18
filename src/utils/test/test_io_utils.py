import unittest
import unittest.mock as mock

import numpy as np

import utils.packets.packet_utils as pack
import utils.io_utils as io_utils

# mock class for utils.event_reading.GtuPdmDataIterator and .AcqL1EventReader
class npyIterator():

    # mock class for utils.event_reading.GtuPdmDataObject
    class frame():

        def __init__(self, photon_count_data):
            self.photon_count_data = photon_count_data

    def __init__(self, packets_list):
        self.packets = packets_list
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
            return npyIterator.frame(self.packets[self.index])
        else:
            raise StopIteration()


class testPacketExtraction(unittest.TestCase):

    # helper methods

    # function to be called every time a packet is extracted
    def _append(self, packet, packet_idx, srcfile):
        self.packets.append(packet)
        self.packet_idxs.append(packet_idx)
        self.srcfiles.append(srcfile)

    # function to reset the lists modified by _append to initial state
    def _reset_lists(self):
        self.packets = []
        self.packet_idxs = []
        self.srcfiles = []

    # assert that the data in the lists is as expected
    def _assertPacketDataAreCorrect(self):
        self.assertEqual(len(self.packets), self.num_packets)
        self.assertEqual(len(self.packet_idxs), self.num_packets)
        self.assertEqual(self.srcfiles, [self.srcfile] * self.num_packets)
        n, w, h = self.template.num_frames, self.template.frame_width, self.template.frame_height
        reference_packet = np.empty((n, w, h))
        for idx in range(self.num_packets):
            self.assertEqual(idx, self.packet_idxs[idx])
            reference_packet.fill(idx)
            self.assertTrue(np.array_equal(self.packets[idx], reference_packet), msg="Packets at index {} are not equal".format(idx))

    # test setup

    def setUp(self):
        # mock filenames
        self.srcfile, self.triggerfile = "srcfile", None

        # the data structures that change when executing the tested functions
        self.packets, self.packet_idxs, self.srcfiles = [], [], []

        # packet template for the functions
        EC_width, EC_height = 16, 16
        width, height, num_frames = 48, 48, 20
        self.template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        self.num_packets = 4
        self.extractor = io_utils.packet_extractor(packet_template=self.template)

        # the list of packets normally loaded from an npy file as a simple numpy array or loaded frame by frame from a ROOT file
        self.bad_packets_num = np.empty((self.num_packets*num_frames + 1, width, height))
        self.bad_packets_width = np.empty((self.num_packets*num_frames, width + 1, height))
        self.bad_packets_height = np.empty((self.num_packets*num_frames, width, height + 1))
        # fill every packet with the value of its index in the list
        self.good_packets_list = np.empty((self.num_packets*num_frames, width, height))
        for idx in range(0, num_frames*self.num_packets, num_frames):
            val = int(idx/num_frames)
            self.good_packets_list[idx:(idx+num_frames)].fill(val)

    # the tests

    @mock.patch('utils.io_utils.np')
    def testNpyfilePacketExtraction(self, mock_np):
        self._reset_lists()
        
        # first test that if the packets within the source are of a different shape from the template, an exception is raised
        mock_np.load.return_value = self.bad_packets_num
        with self.assertRaises(ValueError, msg="Error raising exception for packets source with incorrect number of frames"):
            self.extractor.extract_packets_from_npyfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        mock_np.load.return_value = self.bad_packets_width
        with self.assertRaises(ValueError, msg="Error raising exception for packets source with incorrect frame width"):
            self.extractor.extract_packets_from_npyfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        mock_np.load.return_value = self.bad_packets_height
        with self.assertRaises(ValueError, msg="Error raising exception for packets source with incorrect frame width"):
            self.extractor.extract_packets_from_npyfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        # the function self._append should never be called if the packet template is not valid
        self.assertEqual(len(self.packets), 0)
        self.assertEqual(len(self.packet_idxs), 0)
        self.assertEqual(len(self.srcfiles), 0)

        # now test with a source of packets conforming to the template
        mock_np.load.return_value = self.good_packets_list
        self.extractor.extract_packets_from_npyfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        mock_np.load.assert_called_with(self.srcfile)
        self._assertPacketDataAreCorrect()

    @mock.patch('utils.io_utils.reading')
    def testRootfilePacketExtraction(self, mock_reading):
        self._reset_lists()

        # first test that if the packets within the source are of a different shape from the template, an exception is raised
        mock_reading.AcqL1EventReader.return_value = npyIterator(self.bad_packets_num)
        with self.assertRaises(ValueError, msg="Error raising exception for packets source with incorrect number of frames"):
            self.extractor.extract_packets_from_rootfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        mock_reading.AcqL1EventReader.return_value = npyIterator(self.bad_packets_width)
        with self.assertRaises(ValueError, msg="Error raising exception for packets source with incorrect frame width"):
            self.extractor.extract_packets_from_rootfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        mock_reading.AcqL1EventReader.return_value = npyIterator(self.bad_packets_height)
        with self.assertRaises(ValueError, msg="Error raising exception for packets source with incorrect frame width"):
            self.extractor.extract_packets_from_rootfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        # the function self._append should never be called if the packet template is not valid
        self.assertEqual(len(self.packets), 0)
        self.assertEqual(len(self.packet_idxs), 0)
        self.assertEqual(len(self.srcfiles), 0)

        # now test with a source of packets conforming to the template
        mock_reading.AcqL1EventReader.return_value = npyIterator(self.good_packets_list)
        self.extractor.extract_packets_from_rootfile_and_process(self.srcfile, triggerfile=self.triggerfile, on_packet_extracted=self._append)
        mock_reading.AcqL1EventReader.assert_called_with(self.srcfile, self.triggerfile)
        self._assertPacketDataAreCorrect()


if __name__ == '__main__':
    unittest.main()

