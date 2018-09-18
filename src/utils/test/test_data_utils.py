import unittest
import math
import collections as coll

import numpy as np

import utils.packets.packet_utils as pack
import utils.data_utils as dat

# NOTE: dat.packet_manipulator depends on pack.packet_template, however, it is a relatively simple class 
# and merely provides attributes of a packet and unit conversions for coordinates. The correctness of these 
# features is evaluated by a separate unit test, and since the class does not feature blocking operations, 
# there is not really a need to mock it here, so it is used directly.
# Also, for the same reasons, we are avoiding mocking dat.flat_vals_generator, since it is effectively
# so simple as to not even warrant unit tests (in effect, it technically is a mock already).
class TestPacketManipulator(unittest.TestCase):

    def test_packet_verification(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        manipulator = dat.packet_manipulator(template, verify_against_template=True)
        bad_packet_num = np.empty((num_frames + 10, width, height))
        bad_packet_width = np.empty((num_frames, width + 10, height))
        bad_packet_height = np.empty((num_frames, width, height + 10))
        self.assertRaises(ValueError, manipulator.simu_EC_malfunction, bad_packet_num, 0)
        self.assertRaises(ValueError, manipulator.simu_EC_malfunction, bad_packet_width, 0)
        self.assertRaises(ValueError, manipulator.simu_EC_malfunction, bad_packet_height, 0)
        good_packet = np.empty((num_frames, width, height))
        try:
            manipulator.simu_EC_malfunction(good_packet, 0)
        except ValueError:
            self.fail("manipulator raised ValueError even with a good packet, verification is turned on")
        manipulator.set_packet_verification(False)
        try:
            manipulator.simu_EC_malfunction(good_packet, 0)
        except ValueError:
            self.fail("manipulator raised ValueError even with a good packet, verification is turned off")
        # avoid testing passing bad packets with verification off (programmer using this module is solely responsible for correctness of data passed to it)

    def test_simu_shower(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        manipulator = dat.packet_manipulator(template, verify_against_template=False)
        generator = dat.flat_vals_generator(20, 10)

        angles         = [45, 135, 225, 315, 45, 135, 225, 315]
        start_xs       = [0, width-1, width-1, 0, width-3, 3, 3, width-3]
        start_ys       = [0, 0, height-1, height-1, height-3, height-3, 3, 3]
        start_gtus     = [3, 6, 2, 0, 10, 2, 1, 6]
        durations      = [10, 7, 12, 12, 5, 6, 16, 11]
        num_iterations = [10, 7, 12, 12, 5, 5, 5, 5]
        maximums       = [20, 10, 30, 15, 7, 10, 2, 16]
        ECs            = [[0], [2], [5], [3], [5], [3], [0], [2]]
        num_data       = len(start_xs)
        
        for data_idx in range(num_data):
            packet = np.zeros((num_frames, width, height))
            reference_packet = np.zeros((num_frames, width, height))
            start_x, start_y, start_gtu = start_xs[data_idx], start_ys[data_idx], start_gtus[data_idx]
            start = (start_gtu, start_x, start_y)
            angle, duration, shower_max = math.radians(angles[data_idx]), durations[data_idx], maximums[data_idx]
            delta_x, delta_y = math.cos(angle), math.sin(angle)
            EC_indexes = ECs[data_idx]
            iterations = num_iterations[data_idx]
            
            generator.reset(shower_max, duration)
            ECs_used = manipulator.draw_simulated_shower_line(packet, start, angle, generator)
            
            gtu_idx = start_gtu
            for idx in range(iterations):
                reference_packet[gtu_idx, int(start_x+delta_x*idx), int(start_y+delta_y*idx)] += shower_max
                gtu_idx += 1
            self.assertEqual(np.count_nonzero(packet[start_gtu:gtu_idx]), iterations)
            self.assertTrue(np.array_equal(packet, reference_packet), 
                            msg="Packets at iteration {} are not equal".format(data_idx))
            self.assertEqual(list(coll.Counter(ECs_used).keys()), EC_indexes, 
                            msg="Different number of used ECs at iteration {}".format(data_idx))


    def test_EC_error(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        num_EC = int((width * height) / (EC_width * EC_height))
        shower_ec_indexes = [2, 5]
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        manipulator = dat.packet_manipulator(template, verify_against_template=False)
        packet = np.ones((num_frames, width, height))

        # case 1: method should terminate without changing the packet at all
        manipulator.simu_EC_malfunction(packet, 0, excluded_ECs=shower_ec_indexes)
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, width, height))
        ))
        manipulator.simu_EC_malfunction(packet, 0, excluded_ECs=[])
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, width, height))
        ))

        # case 2: method should only leave untouched the rightmost 2 EC cells
        manipulator.simu_EC_malfunction(packet, num_EC - len(shower_ec_indexes), excluded_ECs=shower_ec_indexes)
        for frame in packet:
            self.assertEqual(np.count_nonzero(frame), EC_width*EC_height*(len(shower_ec_indexes)))
            self.assertTrue(np.array_equal(
                    frame[0:2*EC_width, 0:2*EC_height], np.zeros((width - EC_width, height))
            ))
        
        # case 3: if more malfunctioned ECs are requested than possible without zeroing-out ECs in shower_ec_indexes, then settle for num_EC - len(shower_ec_indexes) 
        packet = np.ones((num_frames, width, height))
        ## only one possible EC can malfunction
        shower_ec_indexes = range(1, num_EC)
        manipulator.simu_EC_malfunction(packet, num_EC, excluded_ECs=shower_ec_indexes)
        for frame in packet:
            self.assertEqual(np.count_nonzero(frame), width*height - EC_width*EC_height)
            self.assertTrue(np.array_equal(
                    frame[0:EC_width, 0:EC_height], np.zeros((EC_width, EC_height))
            ))

if __name__ == '__main__':
    unittest.main()