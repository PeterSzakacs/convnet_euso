import argparse
import unittest

import cmdint.common_args as common_args
import utils.packets.packet_utils as pack

class TestCommonArgs(unittest.TestCase):

    # helper methods

    def _format_input_string_packet(self, base_str, gtu, h, w, ec_h, ec_w):
        return '{} --packet_dims {} {} {} {} {}'.format(base_str, gtu, h, w,
                                                        ec_h, ec_w)

    def _format_input_string_input(self, base_str, in_raw, in_yx, in_gtux,
                                   in_gtuy):
        in_raw = ' --input_raw_packets' if in_raw == True else ''
        in_yx = ' --input_yx_proj' if in_yx == True else ''
        in_gtux = ' --input_gtux_proj' if in_gtux == True else ''
        in_gtuy = ' --input_gtuy_proj' if in_gtuy == True else ''
        return '{}{}{}{}{}'.format(base_str, in_raw, in_yx, in_gtux, in_gtuy)

    def _format_input_string_output(self, base_str, out_raw, out_yx, out_gtux,
                                    out_gtuy):
        out_raw = ' --create_raw_packets' if out_raw == True else ''
        out_yx = ' --create_yx_proj' if out_yx == True else ''
        out_gtux = ' --create_gtux_proj' if out_gtux == True else ''
        out_gtuy = ' --create_gtuy_proj' if out_gtuy == True else ''
        return '{}{}{}{}{}'.format(base_str, out_raw, out_yx,
                                   out_gtux, out_gtuy)

    # test setup

    def setUp(self):
        self._format_types = ('raw', 'yx', 'gtux', 'gtuy')
        self._format_args_parser = argparse.ArgumentParser(description='formats test')
        common_args.add_input_type_dataset_args(self._format_args_parser)
        common_args.add_output_type_dataset_args(self._format_args_parser)

    # common test

    def test_add_cmd_args(self):
        gtu, w, h = 128, 64, 48
        ec_w, ec_h = 32, 16
        i_raw, i_yx, i_gtux, i_gtuy = True, False, True, False
        o_raw, o_yx, o_gtux, o_gtuy = False, True, False, True

        input_str = self._format_input_string_packet('', gtu, h, w, ec_h, ec_w)
        input_str = self._format_input_string_input(input_str, i_raw, i_yx,
                                                    i_gtux, i_gtuy)
        input_str = self._format_input_string_output(input_str, o_raw, o_yx,
                                                    o_gtux, o_gtuy)

        # this code also indiractly tests that the keywords for all arguments
        # do not conflict with each other
        parser = argparse.ArgumentParser(description='test')
        common_args.add_packet_args(parser)
        common_args.add_input_type_dataset_args(parser)
        common_args.add_output_type_dataset_args(parser)

        args = parser.parse_args(input_str.split())
        self.assertListEqual(args.packet_dims, [gtu, h, w, ec_h, ec_w])
        self.assertEqual(args.input_raw_packets, i_raw)
        self.assertEqual(args.input_yx_proj, i_yx)
        self.assertEqual(args.input_gtux_proj, i_gtux)
        self.assertEqual(args.input_gtuy_proj, i_gtuy)
        self.assertEqual(args.create_raw_packets, o_raw)
        self.assertEqual(args.create_yx_proj, o_yx)
        self.assertEqual(args.create_gtux_proj, o_gtux)
        self.assertEqual(args.create_gtuy_proj, o_gtuy)

    # packet arguments test

    def test_packet_args_to_packet_template(self):
        gtu, w, h = 128, 64, 48
        ec_w, ec_h = 32, 16
        str1 = self._format_input_string_packet('', gtu, h, w, ec_h, ec_w)

        parser = argparse.ArgumentParser(description='test')
        common_args.add_packet_args(parser)

        args = parser.parse_args(str1.split())
        template = common_args.packet_args_to_packet_template(args)
        self.assertEqual(template.EC_height, ec_h)
        self.assertEqual(template.EC_width, ec_w)
        self.assertEqual(template.frame_height, h)
        self.assertEqual(template.frame_width, w)
        self.assertEqual(template.num_frames, gtu)

    # input/output data format arguments tests

    def test_format_args_to_dict(self):
        i_raw, i_yx, i_gtux, i_gtuy = True, False, True, False
        input_str = self._format_input_string_input('', i_raw, i_yx,
                                                    i_gtux, i_gtuy)
        o_raw, o_yx, o_gtux, o_gtuy = False, True, False, True
        input_str = self._format_input_string_output(input_str, o_raw, o_yx,
                                                    o_gtux, o_gtuy)

        args = self._format_args_parser.parse_args(input_str.split())
        input_dict = common_args.input_type_dataset_args_to_dict(args)
        output_dict = common_args.output_type_dataset_args_to_dict(args)
        self.assertEqual(input_dict['raw'], i_raw)
        self.assertEqual(input_dict['yx'], i_yx)
        self.assertEqual(input_dict['gtux'], i_gtux)
        self.assertEqual(input_dict['gtuy'], i_gtuy)
        self.assertEqual(output_dict['raw'], o_raw)
        self.assertEqual(output_dict['yx'], o_yx)
        self.assertEqual(output_dict['gtux'], o_gtux)
        self.assertEqual(output_dict['gtuy'], o_gtuy)

    def test_format_args_check(self):
        def test_args_checking(formatter, checker):
            formats_list = [False]*len(self._format_types)
            input_str = formatter('', *formats_list)
            args = self._format_args_parser.parse_args(input_str.split())
            self.assertRaises(Exception, checker, args)

            for idx in range(len(formats_list)):
                formats_list[idx] = True
                input_str = formatter('', *formats_list)
                args = self._format_args_parser.parse_args(input_str.split())
                try:
                    checker(args)
                except Exception:
                    self.fail(msg='Exception raised for valid arguments string {}'.format(input_str))
                formats_list[idx] = False

        test_args_checking(self._format_input_string_input,
                           common_args.check_input_type_dataset_args)
        test_args_checking(self._format_input_string_output,
                           common_args.check_output_type_dataset_args)

    # def test_format_args_to_filenames(self):
    #     ds_name = 'test'
    #     outfiles_ref = tuple('{}_{}.npy'.format(ds_name, _format) for _format in self._format_types)
    #     targetfile_ref = '{}_targets.npy'.format(ds_name)
    #     def test_args_to_filenames(formatter, converter):
    #         formats_list = [True]*len(self._format_types)
    #         input_str = formatter('', *formats_list)
    #         args = self._format_args_parser.parse_args(input_str.split())
    #         outfiles, targetfile = converter(args, ds_name)

    #         self.assertTupleEqual(outfiles_ref, outfiles)
    #         self.assertEqual(targetfile_ref, targetfile)

    #         for idx in range(len(self._format_types)):
    #             formats_list[idx] = False
    #             input_str = formatter('', *formats_list)
    #             args = self._format_args_parser.parse_args(input_str.split())
    #             outfiles, targetfile = converter(args, ds_name)
    #             self.assertTupleEqual(outfiles_ref[idx+1:], outfiles)
    #             self.assertEqual(targetfile_ref, targetfile)
    #     test_args_to_filenames(self._format_input_string_input,
    #                            common_args.input_type_dataset_args_to_filenames)
    #     test_args_to_filenames(self._format_input_string_output,
    #                            common_args.output_type_dataset_args_to_filenames)

if __name__ == '__main__':
    unittest.main()