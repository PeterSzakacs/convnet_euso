import math
import unittest

import numpy as np

import utils.shower_generators as gen
import utils.data_templates as templates
import utils.synth_data_utils as sdutils

# TODO: might want to use custom error message when testing arguments checking

class TestModuleFunctions(unittest.TestCase):

    def _fill_EC_with_zeros(self, packet, X, Y):
        for idx in range(len(X)):
            packet[:, Y[idx], X[idx]] = 0

    def test_simu_shower(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        template = templates.packet_template(EC_width, EC_height, width, height, num_frames)
        generator = gen.flat_vals_generator(20, 10)

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
            start = (start_gtu, start_y, start_x)
            angle, duration, shower_max = angles[data_idx], durations[data_idx], maximums[data_idx]
            iterations = num_iterations[data_idx]

            generator.reset(shower_max, duration)
            GTU, Y, X, vals = sdutils.create_simu_shower_line(angle, start, template, generator)
            packet[GTU, Y, X] += vals

            # create reference data to compare the method call results to
            ang_rad = math.radians(angle)
            delta_x, delta_y = math.cos(ang_rad), math.sin(ang_rad)
            gtu_idx = start_gtu
            ref_X, ref_Y, ref_GTU = [], [], []
            ref_vals = tuple([shower_max,] * iterations)
            for idx in range(iterations):
                y, x = int(start_y+delta_y*idx), int(start_x+delta_x*idx)
                reference_packet[gtu_idx, y, x] += shower_max
                ref_X.append(x); ref_Y.append(y)
                ref_GTU.append(gtu_idx)
                gtu_idx += 1
            self.assertTupleEqual(tuple(ref_X), X)
            self.assertTupleEqual(tuple(ref_Y), Y)
            self.assertTupleEqual(tuple(ref_GTU), GTU)
            self.assertTupleEqual(tuple(ref_vals), vals)
            self.assertTrue(np.array_equal(packet, reference_packet),
                            msg="Packets at iteration {} are not equal".format(data_idx))

    def test_EC_error(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 20
        num_EC = int((width * height) / (EC_width * EC_height))
        shower_ec_indexes = [2, 5]
        template = templates.packet_template(EC_width, EC_height, width, height, num_frames)
        packet = np.ones((num_frames, height, width))

        # case 1: method should terminate without selecting any ECs at all
        X, Y, ECs = sdutils.select_random_ECs(template, 0, excluded_ECs=shower_ec_indexes)
        self._fill_EC_with_zeros(packet, X, Y)
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, height, width))
        ))
        X, Y, ECs = sdutils.select_random_ECs(template, 0, excluded_ECs=[])
        self._fill_EC_with_zeros(packet, X, Y)
        self.assertTrue(np.array_equal(
                packet, np.ones((num_frames, height, width))
        ))

        # case 2: method should select all ECs except the rightmost 2 EC cells
        X, Y, ECs = sdutils.select_random_ECs(template, num_EC - len(shower_ec_indexes), excluded_ECs=shower_ec_indexes)
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
        X, Y, ECs = sdutils.select_random_ECs(template, num_EC, excluded_ECs=shower_ec_indexes)
        self._fill_EC_with_zeros(packet, X, Y)
        for frame in packet:
            self.assertEqual(np.count_nonzero(frame), width*height - EC_width*EC_height)
            self.assertTrue(np.array_equal(
                    frame[0:EC_height, 0:EC_width], np.zeros((EC_height, EC_width))
            ))

if __name__ == '__main__':
    unittest.main()

# Legacy unit test code, partially covers checking of properties in dataset generator
# TODO: fix and put it where it is needed

#     def _format_input_string(self, lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data):
#         input_str = '--bg_lambda {} --packet_dims {} {} {} {} {} --bad_ECs {} {} --num_data {}'
#         return input_str.format(lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data)

#     def testAddCmdArgs(self):
#         lam, gtu, w, h = 0.5, 128, 64, 48
#         ec_w, ec_h, bec_min, bec_max = 32, 16, 0, 3
#         n_data = 10000
#         str1 = self._format_input_string(lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data)

#         args = self.parser.parse_args(str1.split())
#         self.assertListEqual(args.packet_dims, [gtu, h, w, ec_h, ec_w])
#         self.assertEqual(args.bg_lambda, lam)
#         self.assertEqual(args.num_data, n_data)
#         self.assertListEqual(args.bad_ECs, [bec_min, bec_max])

#     def testArgsChecking(self):
#         lam, gtu, w, h = 0.5, 128, 64, 48
#         ec_w, ec_h, bec_min, bec_max = 32, 16, 1, 3
#         n_data = 10000
#         num_ec = int((w*h) / (ec_w*ec_h))

#         # first make sure test does not fail when checking valid args:
#         good_args_str = self._format_input_string(lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data)
#         args = self.parser.parse_args(good_args_str.split())
#         try:
#             self.params_a.check_cmd_args(args)
#         except ValueError:
#             self.fail(msg="Valid arguments did not pass argument check")

#         bad_args = [{'lam': lam, 'gtu': gtu, 'h':h, 'w': w, 'ec_h': ec_h, 'ec_w': ec_w,
#                         'bec_min': bec_min, 'bec_max': bec_max, 'n_data': n_data} for idx in range(6)]
#         # negative lambda
#         bad_args[0]['lam'] = -lam
#         # 0 data items
#         bad_args[1]['n_data'] = 0
#         # negative number of data items passed
#         bad_args[2]['n_data'] = -n_data
#         # negative value for lower bound of number of bad ECs
#         bad_args[3]['bec_min'] = -bec_min
#         # value of upper bound of number of bad ECs is less than the lower bound
#         bad_args[4]['bec_max'] = -bec_max
#         # value of upper bound of number of bad ECs is more than the number of EC modules per frame
#         bad_args[5]['bec_max'] = num_ec + 1
#         for bad_arg in bad_args:
#             str1 = self._format_input_string(**bad_arg)
#             args = self.parser.parse_args(str1.split())
#             self.assertRaises(ValueError, self.params_a.check_cmd_args, args)

#     def _format_input_string(self, sx, sy, sgtu, duration, diff):
#         return '--start_x {} {} --start_gtu {} {} --duration {} {} --bg_diff {} {} --start_y {} {}'.format(
#             sx[0], sx[1], sgtu[0], sgtu[1], duration[0], duration[1], diff[0], diff[1], sy[0], sy[1])

#     def testAddCmdArgs(self):
#         start_x, start_y, start_gtu = [0, 5], [1, 10], [2, 4]
#         duration, bg_diff = [0, 10], [7, 15]

#         input_str = self._format_input_string(start_x, start_y, start_gtu, duration, bg_diff)
#         args = self.parser.parse_args(input_str.split())
#         self.assertListEqual(args.start_x, start_x)
#         self.assertListEqual(args.start_y, start_y)
#         self.assertListEqual(args.start_gtu, start_gtu)
#         self.assertListEqual(args.bg_diff, bg_diff)
#         self.assertListEqual(args.duration, duration)