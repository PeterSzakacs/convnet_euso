import argparse
import collections as coll
import unittest

import cmdint.common_args as cargs
import dataset.constants as cons


class TestPacketArgs(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        cls.long_alias = 'packet'
        cls.packet_args = cargs.packet_args(long_alias=cls.long_alias)

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


class TestDatasetArgs(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        cls.in_alss = {
            'dataset name': 'in_name', 'dataset directory': 'src_dir',
            'dataset': 'in_dset'
        }
        cls.out_alss = {
            'dataset name': 'out_name', 'dataset directory': 'out_dir',
            'dataset': 'out_dset'
        }
        cls.dset_args = cargs.dataset_args(input_aliases=cls.in_alss,
                                            output_aliases=cls.out_alss)

    # test methods
    # dataset single-argument input/output

    def test_add_dataset_arg_single_output(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, cargs.arg_type.OUTPUT)
        alias = self.out_alss['dataset']
        name, dir = 'test', '../dir'
        cmdline = '--{} {}'.format(alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_output_short_alias(self):
        parser = argparse.ArgumentParser()
        alias, s_alias = self.out_alss['dataset'], 'd'
        self.dset_args.add_dataset_arg_single(parser, cargs.arg_type.OUTPUT,
                                              short_alias=s_alias)
        name, dir = 'test', '../dir'
        cmdline = '-{} {}'.format(s_alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_output_multiple(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, cargs.arg_type.OUTPUT,
                                              multiple=True)
        alias = self.out_alss['dataset']
        dsets = [['name1', '../dir2'], ['name2', '../dir1']]
        cmdline = ' '.join('--{} {} {}'.format(alias, dset[0], dset[1])
                           for dset in dsets)
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), dsets)

    def test_add_dataset_arg_single_input(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, cargs.arg_type.INPUT)
        alias = self.in_alss['dataset']
        name, dir = 'test', '../dir'
        cmdline = '--{} {}'.format(alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_input_short_alias(self):
        parser = argparse.ArgumentParser()
        alias, s_alias = self.in_alss['dataset'], 'i'
        self.dset_args.add_dataset_arg_single(parser, cargs.arg_type.INPUT,
                                              short_alias=s_alias)
        name, dir = 'test', '../dir'
        cmdline = '-{} {}'.format(s_alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_input_multiple(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, cargs.arg_type.INPUT,
                                              multiple=True)
        alias = self.in_alss['dataset']
        dsets = [['name1', '../dir2'], ['name2', '../dir1']]
        cmdline = ' '.join('--{} {} {}'.format(alias, dset[0], dset[1])
                           for dset in dsets)
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), dsets)

    # dataset dual-argument input/output (name and dir argument)

    def test_add_dataset_arg_double_output(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.OUTPUT)
        name_als = self.out_alss['dataset name']
        dir_als = self.out_alss['dataset directory']
        name, dset_dir = 'name', 'testdir2/'
        cmdline = '--{} {} --{} {}'.format(name_als, name, dir_als, dset_dir)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, name_als), name)
        self.assertEqual(getattr(args, dir_als), dset_dir)

    def test_add_dataset_arg_double_output_name_alias(self):
        parser = argparse.ArgumentParser()
        name_als, short_als = self.out_alss['dataset name'], 'n'
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.OUTPUT,
                                              name_short_alias=short_als,
                                              required=False)
        name = 'dset_name'
        cmdline = '-{} {}'.format(short_als, name)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, name_als), name)

    def test_add_dataset_arg_double_output_dir_alias(self):
        parser = argparse.ArgumentParser()
        dir_als, short_als = self.out_alss['dataset directory'], 'd'
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.OUTPUT,
                                              dir_short_alias=short_als,
                                              required=False)
        dset_dir = '/var/testdir'
        cmdline = '-{} {}'.format(short_als, dset_dir)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, dir_als), dset_dir)

    def test_add_dataset_arg_double_input(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.INPUT)
        name_als = self.in_alss['dataset name']
        dir_als = self.in_alss['dataset directory']
        name, dset_dir = 'name', 'testdir2/'
        cmdline = '--{} {} --{} {}'.format(name_als, name, dir_als, dset_dir)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, name_als), name)
        self.assertEqual(getattr(args, dir_als), dset_dir)

    def test_add_dataset_arg_double_input_name_alias(self):
        parser = argparse.ArgumentParser()
        name_als, short_als = self.in_alss['dataset name'], 'c'
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.INPUT,
                                              name_short_alias=short_als,
                                              required=False)
        name = 'dset_name'
        cmdline = '-{} {}'.format(short_als, name)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, name_als), name)

    def test_add_dataset_arg_double_input_dir_alias(self):
        parser = argparse.ArgumentParser()
        dir_als, short_als = self.in_alss['dataset directory'], 'i'
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.INPUT,
                                              dir_short_alias=short_als,
                                              required=False)
        dset_dir = '/var/testdir'
        cmdline = '-{} {}'.format(short_als, dset_dir)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, dir_als), dset_dir)

    def test_add_dataset_arg_double_input_dir_missing(self):
        parser = argparse.ArgumentParser()
        name_als, short_als = self.in_alss['dataset name'], 'c'
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.INPUT,
                                              name_short_alias=short_als,
                                              required=True)
        name = 'dset_name'
        cmdline = '-{} {}'.format(short_als, name)
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(cmdline.split())
            self.fail('Failed to raise exception when directory not passed')
        self.assertEqual(cm.exception.code, 2)

    def test_add_dataset_arg_double_input_name_miissing(self):
        parser = argparse.ArgumentParser()
        dir_als, short_als = self.in_alss['dataset directory'], 'i'
        self.dset_args.add_dataset_arg_double(parser, cargs.arg_type.INPUT,
                                              dir_short_alias=short_als,
                                              required=True)
        dset_dir = '/var/testdir'
        cmdline = '-{} {}'.format(short_als, dset_dir)
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(cmdline.split())
            self.fail('Failed to raise exception when name not passed')
        self.assertEqual(cm.exception.code, 2)

    # test get arguments from arg object

    def test_get_dataset_double_input(self):
        attrs = [self.in_alss['dataset name'],
                 self.in_alss['dataset directory']]
        args = coll.namedtuple('args', attrs)('name', '/whatever')
        name, direc = self.dset_args.get_dataset_double(
            args, cargs.arg_type.INPUT)
        self.assertEqual(name, 'name')
        self.assertEqual(direc, '/whatever')

    def test_get_dataset_double_output(self):
        attrs = [self.out_alss['dataset name'],
                 self.out_alss['dataset directory']]
        args = coll.namedtuple('args', attrs)('name', '/whatever')
        name, direc = self.dset_args.get_dataset_double(
            args, cargs.arg_type.OUTPUT)
        self.assertEqual(name, 'name')
        self.assertEqual(direc, '/whatever')

    def test_get_dataset_single_input(self):
        args = coll.namedtuple('args', self.in_alss['dataset'])(
            ['name', '/whatever'])
        dset = self.dset_args.get_dataset_single(args, cargs.arg_type.INPUT)
        self.assertListEqual(dset, ['name', '/whatever'])

    def test_get_dataset_single_output(self):
        args = coll.namedtuple('args', self.out_alss['dataset'])(
            ['name', '/whatever'])
        dset = self.dset_args.get_dataset_single(args, cargs.arg_type.OUTPUT)
        self.assertListEqual(dset, ['name', '/whatever'])


class TestItemTypeArgs(unittest.TestCase):

    # test setup

    def setUp(self):
        self.in_prefix = 'in'
        self.out_prefix = 'out'
        all_types = cons.ALL_ITEM_TYPES
        self.required = {all_types[idx]: (True if idx % 2 == 0 else False)
                         for idx in range(len(all_types))}
        self.item_args = cargs.item_types_args(in_item_prefix=self.in_prefix,
                                               out_item_prefix=self.out_prefix)

    # test methods

    def test_add_item_type_args_input(self):
        parser = argparse.ArgumentParser()
        self.item_args.add_item_type_args(parser, cargs.arg_type.INPUT,
                                          self.required)
        in_pref = self.in_prefix
        exp_attrs = {'{}_{}'.format(in_pref, k): True
                     for k in cons.ALL_ITEM_TYPES}
        exp_args = argparse.Namespace(**exp_attrs)
        cmdline = ['--{}'.format(k) for k in exp_attrs.keys()]
        args = parser.parse_args(cmdline)
        self.assertEqual(args, exp_args)

    def test_add_item_type_args_output(self):
        parser = argparse.ArgumentParser()
        self.item_args.add_item_type_args(parser, cargs.arg_type.OUTPUT,
                                          self.required)
        in_pref = self.out_prefix
        exp_attrs = {'{}_{}'.format(in_pref, k): True
                     for k in cons.ALL_ITEM_TYPES}
        exp_args = argparse.Namespace(**exp_attrs)
        cmdline = ['--{}'.format(k) for k in exp_attrs.keys()]
        args = parser.parse_args(cmdline)
        self.assertEqual(args, exp_args)

    def test_get_item_type_args_input(self):
        attr_names = ['{}_{}'.format(self.in_prefix, item_type)
                      for item_type in cons.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(cons.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        vals = self.item_args.get_item_types(args, cargs.arg_type.INPUT)
        self.assertDictEqual(dict(zip(cons.ALL_ITEM_TYPES, attr_vals)), vals)

    def test_get_item_type_args_output(self):
        attr_names = ['{}_{}'.format(self.out_prefix, item_type)
                      for item_type in cons.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(cons.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        vals = self.item_args.get_item_types(args, cargs.arg_type.OUTPUT)
        self.assertDictEqual(dict(zip(cons.ALL_ITEM_TYPES, attr_vals)), vals)

    def test_check_item_type_args_input_no_exception(self):
        attr_names = ['{}_{}'.format(self.in_prefix, item_type)
                      for item_type in cons.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(cons.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        self.item_args.check_item_type_args(args, cargs.arg_type.INPUT)

    def test_check_item_type_args_output_no_exception(self):
        attr_names = ['{}_{}'.format(self.out_prefix, item_type)
                      for item_type in cons.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(cons.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        self.item_args.check_item_type_args(args, cargs.arg_type.OUTPUT)

    def test_check_item_type_args_input_exception(self):
        attr_names = ['{}_{}'.format(self.in_prefix, item_type)
                      for item_type in cons.ALL_ITEM_TYPES]
        attr_vals = [False] * len(cons.ALL_ITEM_TYPES)
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        try:
            self.item_args.check_item_type_args(args, cargs.arg_type.INPUT)
            self.fail('Item type checking did not throw an error')
        except Exception:
            pass

    def test_check_item_type_args_output_exception(self):
        attr_names = ['{}_{}'.format(self.out_prefix, item_type)
                      for item_type in cons.ALL_ITEM_TYPES]
        attr_vals = [False] * len(cons.ALL_ITEM_TYPES)
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        try:
            self.item_args.check_item_type_args(args, cargs.arg_type.OUTPUT)
            self.fail('Item type checking did not throw an error')
        except Exception:
            pass


class TestMetafieldsOrderArgs(unittest.TestCase):

    def setUp(self):
        self.aliases = {
            'simu': 'simu_test',
            'flight': 'flight_test',
            'synth': 'synth_test'
        }
        self.meta_order_args = cargs.metafield_order_arg(
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
