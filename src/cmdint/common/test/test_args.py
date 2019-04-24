import argparse
import collections as coll
import unittest

import cmdint.common.args as cargs
import dataset.constants as cons


class TestPacketArgs(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        cls.long_alias = 'packet'
        cls.packet_args = cargs.PacketArgs(long_alias=cls.long_alias)

    # test methods

    def test_add_packet_arg_no_alias(self):
        parser = argparse.ArgumentParser()
        packet_dims = [128, 64, 48, 32, 16]
        self.packet_args.add_packet_arg(parser)
        cmdline = '--{} {}'.format(self.long_alias,
                                   ' '.join(str(v) for v in packet_dims))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, self.long_alias), packet_dims)

    def test_add_packet_arg_with_alias(self):
        parser = argparse.ArgumentParser()
        packet_dims = [128, 64, 48, 32, 16]
        short_alias = 'p'
        self.packet_args.add_packet_arg(parser, short_alias=short_alias)
        cmdline = '-{} {}'.format(short_alias,
                                  ' '.join(str(v) for v in packet_dims))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, self.long_alias), packet_dims)

    def test_packet_arg_to_string(self):
        gtu, w, h = 128, 64, 48
        ec_w, ec_h = 32, 16
        args = coll.namedtuple('args', self.long_alias)(
            [gtu, h, w, ec_h, ec_w])
        packet_str = self.packet_args.packet_arg_to_string(args)
        expected_str = 'pack_{}_{}_{}_{}_{}'.format(gtu, h, w, ec_h, ec_w)
        self.assertEqual(packet_str, expected_str)

    def test_packet_arg_to_template(self):
        gtu, w, h = 128, 64, 48
        ec_w, ec_h = 32, 16
        args = coll.namedtuple('args', self.long_alias)(
            [gtu, h, w, ec_h, ec_w])
        template = self.packet_args.packet_arg_to_template(args)
        self.assertEqual(template.EC_height, ec_h)
        self.assertEqual(template.EC_width, ec_w)
        self.assertTupleEqual(template.packet_shape, (gtu, h, w))


class TestMetafieldsOrderArgs(unittest.TestCase):

    def setUp(self):
        self.aliases = {
            'simu': 'simu_test',
            'flight': 'flight_test',
            'synth': 'synth_test'
        }
        self.meta_order_args = cargs.MetafieldOrderArg(
            order_arg_aliases=self.aliases)
        self.parser = argparse.ArgumentParser()
        self.meta_order_args.add_metafields_order_arg(self.parser)

    def test_add_arguments(self):
        cmdline = '--simu_test'.split()
        args = self.parser.parse_args(cmdline)
        exp_args = argparse.Namespace(
            flight_test=None,
            synth_test=None,
            simu_test='simu')
        self.assertEqual(args, exp_args)

    def test_get_metafields_arg(self):
        cmdline = '--simu_test'.split()
        args = self.parser.parse_args(cmdline)
        meta_order = self.meta_order_args.get_metafields_order(args)
        exp_order = cons.METADATA_TYPES['simu']['field order']
        self.assertListEqual(meta_order, exp_order)

    def test_get_metafields_arg_multiple(self):
        cmdline = '--simu_test --flight_test'.split()
        args = self.parser.parse_args(cmdline)
        self.assertRaises(Exception, self.meta_order_args.get_metafields_order,
                          args)

    def test_get_metafields_arg_none_selected(self):
        cmdline = []
        args = self.parser.parse_args(cmdline)
        meta_order = self.meta_order_args.get_metafields_order(
            args, none_selected_ok=True)
        self.assertIsNone(meta_order)


class TestModuleFunctions(unittest.TestCase):

    # test methods

    def test_add_number_range_arg_default(self):
        parser = argparse.ArgumentParser()
        vals = [10, 20]
        cmdline = '--foo {} {}'.format(vals[0], vals[1])
        cargs.add_number_range_arg(parser, 'foo')
        args = parser.parse_args(cmdline.split())
        self.assertEqual(args.foo, vals)

    def test_add_number_range_arg_short_alias(self):
        parser = argparse.ArgumentParser()
        vals = [10, 20]
        cmdline = '-f {} {}'.format(vals[0], vals[1])
        cargs.add_number_range_arg(parser, 'foo', short_alias='f')
        args = parser.parse_args(cmdline.split())
        self.assertEqual(args.foo, vals)


if __name__ == '__main__':
    unittest.main()
