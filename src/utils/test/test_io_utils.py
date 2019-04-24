import os
import unittest
import unittest.mock as mock

import numpy as np
import numpy.testing as nptest

import test.test_setups as testset
import utils.data_templates as templates
import utils.io_utils as io_utils


@mock.patch('os.path.exists')
@mock.patch('os.path.isfile')
class TestModuleFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filename = 'test.tsv'
        cls.m_rows = [{'testkey': 'testvar', 'otherkey': 'otherval'}]
        cls.cols_order = ('testkey', 'otherkey')
        cls.file_contents = 'testkey\totherkey\r\ntestvar\totherval\r\n'

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Save_TSV(self, m_open, m_isfile, m_exists):
        m_exists.return_value = m_isfile.return_value = False
        buf = testset.MockTextFileStream()
        m_open.return_value = buf
        io_utils.save_TSV(self.filename, self.m_rows,
                          self.cols_order)
        m_open.assert_called_once_with(self.filename, 'w', encoding='UTF-8')
        self.assertEqual(buf.temp_buf, self.file_contents)

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Save_TSV_overwrite_existing_file(self, m_open, m_isfile,
                                              m_exists):
        m_exists.return_value = m_isfile.return_value = True
        buf = testset.MockTextFileStream(initial_value='foo')
        m_open.return_value = buf
        io_utils.save_TSV(self.filename, self.m_rows, self.cols_order,
                            file_exists_overwrite=True)
        m_open.assert_called_once_with(self.filename, 'w', encoding='UTF-8')
        self.assertEqual(buf.temp_buf, self.file_contents)

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Save_TSV_do_not_overwrite_existing_file(self, m_open, m_isfile,
                                                     m_exists):
        m_exists.return_value = m_isfile.return_value = True
        self.assertRaises(FileExistsError, io_utils.save_TSV, self.filename,
                          self.m_rows, self.cols_order,
                          file_exists_overwrite=False)
        m_open.assert_not_called()

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_Load_TSV(self, m_open, m_isfile, m_exists):
        m_exists.return_value = m_isfile.return_value = True
        m_open.return_value = io.StringIO(self.file_contents)
        rows = io_utils.load_TSV(self.filename)
        self.assertEqual(rows, self.m_rows)
        m_open.assert_called_with(self.filename, 'r', encoding='UTF-8')


class TestPacketExtractor(unittest.TestCase):

    # mock class for utils.event_reading.GtuPdmDataIterator
    # and utils.event_reading.AcqL1EventReader
    class NpyIterator():

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
                return TestPacketExtractor.NpyIterator.frame(
                    self.index,
                    self.packets[self.index].reshape(self.new_shape)
                )
            else:
                raise StopIteration()

    # test setup

    @classmethod
    def setUpClass(cls):
        # mock filenames
        cls.srcfile, cls.triggerfile = "srcfile", None
        # packet template for the functions
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        cls.template = templates.packet_template(EC_width, EC_height,
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
        cls.extractor = io_utils.PacketExtractor(
            packet_template=cls.template)

    # test methods

    @mock.patch('numpy.load')
    def test_extract_from_npyfile(self, m_np_load):
        m_np_load.return_value = self.packets
        extracted_packets = self.extractor.extract_packets_from_npyfile(
            self.srcfile, self.triggerfile)
        m_np_load.assert_called_with(self.srcfile)
        nptest.assert_array_equal(extracted_packets, self.expected_packets)

    @mock.patch('utils.io_utils.reading')
    def test_extract_from_rootfile(self, m_reading):
        it = TestPacketExtractor.NpyIterator(self.packets)
        m_reading.AcqL1EventReader.return_value = it
        extracted_packets = self.extractor.extract_packets_from_rootfile(
            self.srcfile, self.triggerfile)
        m_reading.AcqL1EventReader.assert_called_with(self.srcfile,
                                                         self.triggerfile)
        nptest.assert_array_equal(extracted_packets, self.expected_packets)


if __name__ == '__main__':
    unittest.main()
