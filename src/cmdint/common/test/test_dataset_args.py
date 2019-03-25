import argparse
import collections as coll
import unittest

import cmdint.common.dataset_args as dargs
import dataset.constants as cons


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
        cls.dset_args = dargs.dataset_args(input_aliases=cls.in_alss,
                                            output_aliases=cls.out_alss)

    # test methods
    # dataset single-argument input/output

    def test_add_dataset_arg_single_output(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, dargs.arg_type.OUTPUT)
        alias = self.out_alss['dataset']
        name, dir = 'test', '../dir'
        cmdline = '--{} {}'.format(alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_output_short_alias(self):
        parser = argparse.ArgumentParser()
        alias, s_alias = self.out_alss['dataset'], 'd'
        self.dset_args.add_dataset_arg_single(parser, dargs.arg_type.OUTPUT,
                                              short_alias=s_alias)
        name, dir = 'test', '../dir'
        cmdline = '-{} {}'.format(s_alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_output_multiple(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, dargs.arg_type.OUTPUT,
                                              multiple=True)
        alias = self.out_alss['dataset']
        dsets = [['name1', '../dir2'], ['name2', '../dir1']]
        cmdline = ' '.join('--{} {} {}'.format(alias, dset[0], dset[1])
                           for dset in dsets)
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), dsets)

    def test_add_dataset_arg_single_input(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, dargs.arg_type.INPUT)
        alias = self.in_alss['dataset']
        name, dir = 'test', '../dir'
        cmdline = '--{} {}'.format(alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_input_short_alias(self):
        parser = argparse.ArgumentParser()
        alias, s_alias = self.in_alss['dataset'], 'i'
        self.dset_args.add_dataset_arg_single(parser, dargs.arg_type.INPUT,
                                              short_alias=s_alias)
        name, dir = 'test', '../dir'
        cmdline = '-{} {}'.format(s_alias, ' '.join([name, dir]))
        args = parser.parse_args(cmdline.split())
        self.assertListEqual(getattr(args, alias), [name, dir])

    def test_add_dataset_arg_single_input_multiple(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_single(parser, dargs.arg_type.INPUT,
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
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.OUTPUT)
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
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.OUTPUT,
                                              name_short_alias=short_als,
                                              required=False)
        name = 'dset_name'
        cmdline = '-{} {}'.format(short_als, name)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, name_als), name)

    def test_add_dataset_arg_double_output_dir_alias(self):
        parser = argparse.ArgumentParser()
        dir_als, short_als = self.out_alss['dataset directory'], 'd'
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.OUTPUT,
                                              dir_short_alias=short_als,
                                              required=False)
        dset_dir = '/var/testdir'
        cmdline = '-{} {}'.format(short_als, dset_dir)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, dir_als), dset_dir)

    def test_add_dataset_arg_double_input(self):
        parser = argparse.ArgumentParser()
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.INPUT)
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
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.INPUT,
                                              name_short_alias=short_als,
                                              required=False)
        name = 'dset_name'
        cmdline = '-{} {}'.format(short_als, name)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, name_als), name)

    def test_add_dataset_arg_double_input_dir_alias(self):
        parser = argparse.ArgumentParser()
        dir_als, short_als = self.in_alss['dataset directory'], 'i'
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.INPUT,
                                              dir_short_alias=short_als,
                                              required=False)
        dset_dir = '/var/testdir'
        cmdline = '-{} {}'.format(short_als, dset_dir)
        args = parser.parse_args(cmdline.split())
        self.assertEqual(getattr(args, dir_als), dset_dir)

    def test_add_dataset_arg_double_input_dir_missing(self):
        parser = argparse.ArgumentParser()
        name_als, short_als = self.in_alss['dataset name'], 'c'
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.INPUT,
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
        self.dset_args.add_dataset_arg_double(parser, dargs.arg_type.INPUT,
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
            args, dargs.arg_type.INPUT)
        self.assertEqual(name, 'name')
        self.assertEqual(direc, '/whatever')

    def test_get_dataset_double_output(self):
        attrs = [self.out_alss['dataset name'],
                 self.out_alss['dataset directory']]
        args = coll.namedtuple('args', attrs)('name', '/whatever')
        name, direc = self.dset_args.get_dataset_double(
            args, dargs.arg_type.OUTPUT)
        self.assertEqual(name, 'name')
        self.assertEqual(direc, '/whatever')

    def test_get_dataset_single_input(self):
        args = coll.namedtuple('args', self.in_alss['dataset'])(
            ['name', '/whatever'])
        dset = self.dset_args.get_dataset_single(args, dargs.arg_type.INPUT)
        self.assertListEqual(dset, ['name', '/whatever'])

    def test_get_dataset_single_output(self):
        args = coll.namedtuple('args', self.out_alss['dataset'])(
            ['name', '/whatever'])
        dset = self.dset_args.get_dataset_single(args, dargs.arg_type.OUTPUT)
        self.assertListEqual(dset, ['name', '/whatever'])


class TestItemTypeArgs(unittest.TestCase):

    # test setup

    def setUp(self):
        self.in_prefix = 'in'
        self.out_prefix = 'out'
        all_types = cons.ALL_ITEM_TYPES
        self.required = {all_types[idx]: (True if idx % 2 == 0 else False)
                         for idx in range(len(all_types))}
        self.item_args = dargs.item_types_args(in_item_prefix=self.in_prefix,
                                               out_item_prefix=self.out_prefix)

    # test methods

    def test_add_item_type_args_input(self):
        parser = argparse.ArgumentParser()
        self.item_args.add_item_type_args(parser, dargs.arg_type.INPUT,
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
        self.item_args.add_item_type_args(parser, dargs.arg_type.OUTPUT,
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
        vals = self.item_args.get_item_types(args, dargs.arg_type.INPUT)
        self.assertDictEqual(dict(zip(cons.ALL_ITEM_TYPES, attr_vals)), vals)

    def test_get_item_type_args_output(self):
        attr_names = ['{}_{}'.format(self.out_prefix, item_type)
                      for item_type in cons.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(cons.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        vals = self.item_args.get_item_types(args, dargs.arg_type.OUTPUT)
        self.assertDictEqual(dict(zip(cons.ALL_ITEM_TYPES, attr_vals)), vals)


if __name__ == '__main__':
    unittest.main()
