import unittest
import math

import numpy as np
np.set_printoptions(threshold=np.nan)

import utils.packets.packet_utils as pack
import utils.data_utils as dat

# NOTE: dat.packet_manipulator depends on pack.packet_template, however, it is a relatively simple class 
# and merely provides attributes of a packet and unit conversions for coordinates. The correctness of these 
# features is evaluated by a separate unit test, and since the class does not feature blocking operations, 
# there is not really a need to mock it here, so it is used directly.
# Also, for the same reasons, we are avoiding mocking dat.flat_vals_generator, since it is effectively
# so simple as to not even warrant unit tests (in effect, it technically is a mock already).
class TestPacketManipulator(unittest.TestCase):

    # deprecated test: except for projections, packet_manipulator does not really check packets anymore, 
    # and those methods probably don't belong here anyway
    def test_packet_verification(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        manipulator = dat.packet_manipulator(template, verify_against_template=True)
        bad_packet_num = np.empty((num_frames + 10, height, width))
        bad_packet_height = np.empty((num_frames, height + 10, width))
        bad_packet_width = np.empty((num_frames, height, width + 10))
        self.assertRaises(ValueError, manipulator.create_x_y_projection, bad_packet_num)
        self.assertRaises(ValueError, manipulator.create_x_y_projection, bad_packet_height)
        self.assertRaises(ValueError, manipulator.create_x_y_projection, bad_packet_width)
        good_packet = np.empty((num_frames, height, width))
        try:
            manipulator.create_x_y_projection(good_packet)
        except ValueError:
            self.fail("manipulator raised ValueError even with a good packet, verification is turned on")
        manipulator.set_packet_verification(False)
        try:
            manipulator.create_x_y_projection(good_packet)
        except ValueError:
            self.fail("manipulator raised ValueError even with a good packet, verification is turned off")
        #avoid testing passing bad packets with verification off (programmer using this module is solely responsible for correctness of data passed to it)

    # testing the returning of a list of used ECs is just commented out instead of removed, 
    # in case we want to return this feature later
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
        num_data       = len(start_xs)
        
        for data_idx in range(num_data):
            packet = np.ones((num_frames, height, width))
            reference_packet = np.ones((num_frames, height, width))
            start_x, start_y, start_gtu = start_xs[data_idx], start_ys[data_idx], start_gtus[data_idx]
            start = (start_gtu, start_x, start_y)
            angle, duration, shower_max = math.radians(angles[data_idx]), durations[data_idx], maximums[data_idx]
            delta_x, delta_y = math.cos(angle), math.sin(angle)
            iterations = num_iterations[data_idx]
            
            generator.reset(shower_max, duration)
            X, Y, GTU, vals = manipulator.draw_simulated_shower_line(start, angle, generator)
            packet[GTU, Y, X] += vals

            # create reference data to compare the method call results to
            gtu_idx = start_gtu
            ref_X, ref_Y, ref_GTU = [], [], []
            # ref_EC = []
            ref_vals = [shower_max,] * iterations
            for idx in range(iterations):
                y, x = int(start_y+delta_y*idx), int(start_x+delta_x*idx)
                reference_packet[gtu_idx, y, x] += shower_max
                ref_X.append(x); ref_Y.append(y)
                ref_GTU.append(gtu_idx)
                #ref_EC.append(template.xy_to_ec_idx(x, y))
                gtu_idx += 1
            self.assertTupleEqual(tuple(ref_X), X)
            self.assertTupleEqual(tuple(ref_Y), Y)
            self.assertTupleEqual(tuple(ref_GTU), GTU)
            self.assertTupleEqual(tuple(ref_vals), vals)
            #self.assertTupleEqual(tuple(ref_EC), ECs_used)
            self.assertTrue(np.array_equal(packet, reference_packet), 
                            msg="Packets at iteration {} are not equal".format(data_idx))

    def test_EC_error(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        num_EC = int((width * height) / (EC_width * EC_height))
        shower_ec_indexes = [2, 5]
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        manipulator = dat.packet_manipulator(template, verify_against_template=False)
        packet = np.ones((num_frames, height, width))

        # case 1: method should terminate without selecting any ECs at all
        X, Y, ECs = manipulator.select_random_ECs(0, excluded_ECs=shower_ec_indexes)
        self._fill_EC_with_zeros(packet, X, Y)
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, height, width))
        ))
        X, Y, ECs = manipulator.select_random_ECs(0, excluded_ECs=[])
        self._fill_EC_with_zeros(packet, X, Y)
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, height, width))
        ))

        # case 2: method should select all ECs except the rightmost 2 EC cells
        X, Y, ECs = manipulator.select_random_ECs(num_EC - len(shower_ec_indexes), excluded_ECs=shower_ec_indexes)
        self._fill_EC_with_zeros(packet, X, Y)
        for frame in packet:
            self.assertEqual(np.count_nonzero(frame), EC_width*EC_height*(len(shower_ec_indexes)))
            self.assertTrue(np.array_equal(
                    frame[0:2*EC_height, 0:2*EC_width], np.zeros((height, width - EC_width))
            ))
        
        # case 3: if more malfunctioned ECs are requested than possible without selecting ECs in excluded_ECs, then settle for num_EC - len(excluded_ECs)
        packet = np.ones((num_frames, width, height))
        ## only one possible EC can malfunction
        shower_ec_indexes = range(1, num_EC)
        X, Y, ECs = manipulator.select_random_ECs(num_EC, excluded_ECs=shower_ec_indexes)
        self._fill_EC_with_zeros(packet, X, Y)
        for frame in packet:
            self.assertEqual(np.count_nonzero(frame), width*height - EC_width*EC_height)
            self.assertTrue(np.array_equal(
                    frame[0:EC_height, 0:EC_width], np.zeros((EC_height, EC_width))
            ))

    def _fill_EC_with_zeros(self, packet, X, Y):
        for idx in range(len(X)):
            packet[:, Y[idx], X[idx]] = 0

    # deprecated test: these methods should probably be moved into a different module
    def test_projections(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        manipulator = dat.packet_manipulator(template, verify_against_template=False)
        packet = np.ones((num_frames, height, width))
        self.assertTupleEqual(manipulator.create_x_y_projection(packet).shape, (height, width))
        self.assertTupleEqual(manipulator.create_x_gtu_projection(packet).shape, (num_frames, width))
        self.assertTupleEqual(manipulator.create_y_gtu_projection(packet).shape, (num_frames, height))


if __name__ == '__main__':
    unittest.main()