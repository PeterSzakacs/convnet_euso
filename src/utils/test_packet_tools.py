import unittest
import math
import collections as coll

import numpy as np

import packet_tools as pack

# Note: EC height is delibarately unrealistic and different from EC_width for better testing and debugging
class TestPacketManipulator(unittest.TestCase):

    def test_visible_properties(self):
        EC_width, EC_height = 16, 32
        width, height = 48, 64
        num_EC = (width * height) / (EC_width * EC_height)
        manipulator = pack.packet_manipulator(EC_width, EC_height, width, height)
        self.assertEqual(EC_width, manipulator.EC_width)
        self.assertEqual(EC_height, manipulator.EC_height)
        self.assertEqual(width, manipulator.frame_width)
        self.assertEqual(height, manipulator.frame_height)
        self.assertEqual(int(width/EC_width), manipulator.num_cols)
        self.assertEqual(int(height/EC_height), manipulator.num_rows)
        self.assertEqual(num_EC, manipulator.num_EC)


    def test_unit_conversions(self):
        EC_width, EC_height = 16, 32
        width, height = 48, 64
        manipulator = pack.packet_manipulator(EC_width, EC_height, width, height)
        xs      = [0,  10, 21, 30, 21, 40, 0]
        ys      = [10, 0,  1,  42, 10, 50, 0]
        ec_ids  = [0,  0,  1,  4,  1,  5,  0]
        ec_xys  = [[0, 0], [0, 0], [1, 0], [1, 1], [1, 0], [2, 1], [0, 0]]
        for idx in range(len(xs)):
            x, y, ec_x, ec_y, ec_idx = xs[idx], ys[idx], ec_xys[idx][0], ec_xys[idx][1], ec_ids[idx]
            self.assertEqual(manipulator.x_to_ec_x(x), ec_x)
            self.assertEqual(manipulator.y_to_ec_y(y), ec_y)
            self.assertEqual(manipulator.ec_idx_to_ec_xy(ec_idx), (ec_x, ec_y))
            self.assertEqual(manipulator.xy_to_ec_idx(x, y), ec_idx)
            self.assertEqual(manipulator.ec_xy_to_ec_idx(ec_x, ec_y), ec_idx)
            self.assertEqual(manipulator.ec_idx_to_ec_xy(ec_idx), (ec_x, ec_y))

    def test_simu_shower(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        manipulator = pack.packet_manipulator(EC_width, EC_height, width, height)
        generator = pack.flat_vals_generator(20, 10)

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
            angle, duration, shower_max = math.radians(angles[data_idx]), durations[data_idx], maximums[data_idx]
            delta_x, delta_y = math.cos(angle), math.sin(angle)
            EC_indexes = ECs[data_idx]
            iterations = num_iterations[data_idx]
            
            generator.reset(shower_max, duration)
            ECs_used = manipulator.draw_simulated_shower_line(packet, start_x, start_y, angle, generator, start_gtu=start_gtu)
            
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
        width, height = 48, 64
        num_frames = 20
        num_EC = int((width * height) / (EC_width * EC_height))
        shower_ec_indexes = [2, 5]
        manipulator = pack.packet_manipulator(EC_width, EC_height, width, height)
        packet = np.ones((num_frames, width, height))

        # case 1: method should terminate without changing the packet at all
        manipulator.simu_EC_malfunction(packet, 0, shower_EC_indexes=shower_ec_indexes)
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, width, height))
        ))
        manipulator.simu_EC_malfunction(packet, 0, shower_EC_indexes=[])
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, width, height))
        ))

        # case 2: method should only leave untouched the rightmost 2 EC cells
        manipulator.simu_EC_malfunction(packet, num_EC - len(shower_ec_indexes), shower_EC_indexes=shower_ec_indexes)
        for frame in packet:
            self.assertEqual(np.count_nonzero(frame), EC_width*EC_height*(len(shower_ec_indexes)))
            self.assertTrue(np.array_equal(
                    frame[0:2*EC_width, 0:2*EC_height], np.zeros((width - EC_width, height))
            ))
        
        # case 3: if more malfunctioned ECs are requested than possible without zeroing-out ECs in shower_ec_indexes, then settle for num_EC - len(shower_ec_indexes) 
        packet = np.ones((num_frames, width, height))
        ## only one possible EC can malfunction
        shower_ec_indexes = range(1, num_EC)
        manipulator.simu_EC_malfunction(packet, num_EC, shower_EC_indexes=shower_ec_indexes)
        for frame in packet:
            self.assertEqual(np.count_nonzero(frame), width*height - EC_width*EC_height)
            self.assertTrue(np.array_equal(
                    frame[0:EC_width, 0:EC_height], np.zeros((EC_width, EC_height))
            ))

if __name__ == '__main__':
    unittest.main()