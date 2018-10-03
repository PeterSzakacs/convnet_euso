import argparse
import unittest

import utils.packets.packet_utils as pack
import utils.synth_data_utils as sdutils

# TODO: might want to use custom error message when testing arguments checking

class TestParamsArgs(unittest.TestCase):

    def _format_input_string(self, lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data):
        input_str = '--bg_lambda {} --packet_dims {} {} {} {} {} --bad_ECs {} {} --num_data {}'
        return input_str.format(lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data)

    def setUp(self):
        self.parser = argparse.ArgumentParser(description="test")
        self.params_a = sdutils.params_args()
        self.params_a.add_packet_cmd_args(self.parser)
        self.params_a.add_other_cmd_args(self.parser)

    def testAddCmdArgs(self):
        lam, gtu, w, h = 0.5, 128, 64, 48
        ec_w, ec_h, bec_min, bec_max = 32, 16, 0, 3
        n_data = 10000
        str1 = self._format_input_string(lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data)        

        args = self.parser.parse_args(str1.split())
        self.assertListEqual(args.packet_dims, [gtu, h, w, ec_h, ec_w])
        self.assertEqual(args.bg_lambda, lam)
        self.assertEqual(args.num_data, n_data)
        self.assertListEqual(args.bad_ECs, [bec_min, bec_max])

    def testArgsToPacketTemplate(self):
        lam, gtu, w, h = 0.5, 128, 64, 48
        ec_w, ec_h, bec_min, bec_max = 32, 16, 0, 3
        n_data = 10000
        str1 = self._format_input_string(lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data)

        args = self.parser.parse_args(str1.split())
        template = self.params_a.args_to_packet_template(args)
        self.assertEqual(template.EC_height, ec_h)
        self.assertEqual(template.EC_width, ec_w)
        self.assertEqual(template.frame_height, h)
        self.assertEqual(template.frame_width, w)
        self.assertEqual(template.num_frames, gtu)

    # NOTE: packet template properties checking is implemented in packet_utils and covered by its unit test
    def testArgsChecking(self):
        lam, gtu, w, h = 0.5, 128, 64, 48
        ec_w, ec_h, bec_min, bec_max = 32, 16, 1, 3
        n_data = 10000
        num_ec = int((w*h) / (ec_w*ec_h))

        # first make sure test does not fail when checking valid args:
        good_args_str = self._format_input_string(lam, gtu, h, w, ec_h, ec_w, bec_min, bec_max, n_data)
        args = self.parser.parse_args(good_args_str.split())
        try:
            self.params_a.check_cmd_args(args)
        except ValueError:
            self.fail(msg="Valid arguments did not pass argument check")

        bad_args = [{'lam': lam, 'gtu': gtu, 'h':h, 'w': w, 'ec_h': ec_h, 'ec_w': ec_w, 
                        'bec_min': bec_min, 'bec_max': bec_max, 'n_data': n_data} for idx in range(6)]
        # negative lambda
        bad_args[0]['lam'] = -lam
        # 0 data items
        bad_args[1]['n_data'] = 0
        # negative number of data items passed
        bad_args[2]['n_data'] = -n_data
        # negative value for lower bound of number of bad ECs
        bad_args[3]['bec_min'] = -bec_min
        # value of upper bound of number of bad ECs is less than the lower bound
        bad_args[4]['bec_max'] = -bec_max
        # value of upper bound of number of bad ECs is more than the number of EC modules per frame
        bad_args[5]['bec_max'] = num_ec + 1
        for bad_arg in bad_args:
            str1 = self._format_input_string(**bad_arg)
            args = self.parser.parse_args(str1.split())
            self.assertRaises(ValueError, self.params_a.check_cmd_args, args)



class TestShowerArgs(unittest.TestCase):

    def _format_input_string(self, sx, sy, sgtu, duration, diff):
        return '--start_x {} {} --start_gtu {} {} --duration {} {} --bg_diff {} {} --start_y {} {}'.format(
            sx[0], sx[1], sgtu[0], sgtu[1], duration[0], duration[1], diff[0], diff[1], sy[0], sy[1])

    def setUp(self):
        self.parser = argparse.ArgumentParser(description='test')
        self.shower_a = sdutils.shower_args()
        self.shower_a.add_cmd_args(self.parser)

    def testAddCmdArgs(self):
        start_x, start_y, start_gtu = [0, 5], [1, 10], [2, 4]
        duration, bg_diff = [0, 10], [7, 15]

        input_str = self._format_input_string(start_x, start_y, start_gtu, duration, bg_diff)
        args = self.parser.parse_args(input_str.split())
        self.assertListEqual(args.start_x, start_x)
        self.assertListEqual(args.start_y, start_y)
        self.assertListEqual(args.start_gtu, start_gtu)
        self.assertListEqual(args.bg_diff, bg_diff)
        self.assertListEqual(args.duration, duration)

    def testArgsToDict(self):
        start_x, start_y, start_gtu = [0, 5], [1, 10], [2, 4]
        duration, bg_diff = [0, 10], [7, 15]
        
        input_str = self._format_input_string(start_x, start_y, start_gtu, duration, bg_diff)
        args = self.parser.parse_args(input_str.split())
        args_dictionary = self.shower_a.args_to_dict(args)
        ref_dictionary = {'start_x': start_x, 'start_y': start_y, 'start_gtu': start_gtu, 
                            'duration': duration, 'bg_diff': bg_diff}
        
        self.assertDictEqual(args_dictionary, ref_dictionary)

    def testArgsChecking(self):
        gtu, w, h, ec_w, ec_h = 128, 64, 48, 32, 16
        template = pack.packet_template(ec_w, ec_h, w, h, gtu)
        start_x, start_y, start_gtu = (3, 5), (1, 10), (2, 4)
        duration, bg_diff = (2, 10), (7, 15)

        # first make sure test does not fail when checking valid args:
        good_args_str = self._format_input_string(start_x, start_y, start_gtu, duration, bg_diff)
        args = self.parser.parse_args(good_args_str.split())
        try:
            self.shower_a.check_cmd_args(args, template)
        except ValueError:
            self.fail(msg="Valid arguments did not pass argument check")

        bad_args = [{'sx': start_x, 'sy': start_y, 'sgtu': start_gtu, 'duration': duration, 'diff': bg_diff} for idx in range(14)]
        # start_x lower bound is less than 0, upper bound is less than lower bound or upper bound is larger than frame width
        bad_args[0]['sx'] = (-start_x[0], start_x[1])
        bad_args[1]['sx'] = (start_x[1], start_x[0])
        bad_args[2]['sx'] = (start_x[0], w+1)
        # start_y lower bound is less than 0, upper bound is less than lower bound or upper bound is larger than frame height
        bad_args[3]['sy'] = (-start_y[0], start_y[1])
        bad_args[4]['sy'] = (start_y[1], start_y[0])
        bad_args[5]['sy'] = (start_y[0], h+1)
        # start_gtu lower bound is less than 0, upper bound is less than lower bound or upper bound is larger than number of frames
        bad_args[6]['sgtu'] = (-start_gtu[0], start_gtu[1])
        bad_args[7]['sgtu'] = (start_gtu[1], start_gtu[0])
        bad_args[8]['sgtu'] = (start_gtu[0], gtu+1)
        # duration lower bound is less than 1, upper bound is less than lower bound or upper bound is larger than the number of frames
        bad_args[9]['duration'] = (0, duration[1])
        bad_args[10]['duration'] = (duration[1], duration[0])
        bad_args[11]['duration'] = (duration[0], gtu+1)
        # bg_diff lower bound is less than 1 or upper bound is less than lower bound
        bad_args[12]['diff'] = (0, bg_diff[1])
        bad_args[13]['diff'] = (bg_diff[1], bg_diff[0])

        for bad_arg in bad_args:
            input_str = self._format_input_string(**bad_arg)
            args = self.parser.parse_args(input_str.split())
            self.assertRaises(ValueError, self.shower_a.check_cmd_args, args, template)
    
    # def testArgsToString(self):
    #     gtu, w, h, ec_w, ec_h = 128, 64, 48, 32, 16
    #     template = pack.packet_template(ec_w, ec_h, w, h, gtu)
    #     start_x, start_y, start_gtu = (3, 5), (1, 10), (6, 8)
    #     duration, bg_diff = (2, 11), (7, 15)

    #     # first make sure test does not fail when checking valid args:
    #     input_str = self._format_input_string(start_x, start_y, start_gtu, duration, bg_diff)
    #     args = self.parser.parse_args(good_args_str.split())
    #     result = self.shower_a.args_to_string(args)
    #     self.assertEqual(result, 'shower')


if __name__ == '__main__':
    unittest.main()