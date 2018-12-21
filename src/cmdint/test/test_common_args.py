import collections as coll
import unittest
import unittest.mock as mock

import utils.dataset_utils as ds
import cmdint.common_args as cargs


class TestPacketArgs(unittest.TestCase):

    # test setup

    def setUp(self):
        self.long_alias = 'packet'
        self.packet_args = cargs.packet_args(long_alias=self.long_alias)

    # test methods

    def test_add_packet_arg_no_alias(self):
        mock_parser = mock.MagicMock()
        expected_aliases = ('--{}'.format(self.long_alias), )
        self.packet_args.add_packet_arg(mock_parser)
        self.assertEqual(mock_parser.add_argument.call_count, 1)
        self.assertTupleEqual(mock_parser.add_argument.call_args[0],
                              expected_aliases)

    def test_add_packet_arg_with_alias(self):
        mock_parser = mock.MagicMock()
        expected_aliases = ('-p', '--{}'.format(self.long_alias))
        self.packet_args.add_packet_arg(mock_parser, short_alias='p')
        self.assertEqual(mock_parser.add_argument.call_count, 1)
        self.assertTupleEqual(mock_parser.add_argument.call_args[0],
                              expected_aliases)

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

    # helper methods (custom assert)

    def _assert_call_single(self, mock_parser, exp_pos, action, atype,
                            multiple=False):
        self.assertEqual(mock_parser.add_argument.call_count, 1)
        self.assertEqual(mock_parser.add_argument.call_args[0], exp_pos)
        kw = mock_parser.add_argument.call_args[1]
        metavar = self.i_meta if atype == cargs.arg_type.INPUT else self.o_meta
        help = '{} dataset'.format(atype.value)
        if multiple:
            help += '(s)'
        self.assertEqual(kw['metavar'], metavar)
        self.assertEqual(kw['action'], action)
        self.assertEqual(kw['help'], help)

    def _assert_call_double(self, mock_parser, exp_name_pos, exp_dir_pos,
                            atype):
        self.assertEqual(mock_parser.add_argument.call_count, 2)
        name_pos, name_kw = mock_parser.add_argument.call_args_list[0][:]
        dir_pos, dir_kw = mock_parser.add_argument.call_args_list[1][:]
        self.assertEqual(name_pos, exp_name_pos)
        self.assertEqual(dir_pos, exp_dir_pos)
        dir_pos, dir_kw = mock_parser.add_argument.call_args_list[1][:]
        v = atype.value
        self.assertEqual(name_kw['help'], '{} dataset name'.format(v))
        self.assertEqual(dir_kw['help'], '{} dataset directory'.format(v))

    # test setup

    def setUp(self):
        self.i_meta = ('NAME', 'INPUT_DIR')
        self.o_meta = ('NAME', 'OUTPUT_DIR')
        self.in_alss = {
            'dataset name': 'in_name', 'dataset directory': 'src_dir',
            'dataset': 'in_dset'
        }
        self.out_alss = {
            'dataset name': 'out_name', 'dataset directory': 'out_dir',
            'dataset': 'out_dset'
        }
        self.dset_args = cargs.dataset_args(input_aliases=self.in_alss,
                                            output_aliases=self.out_alss)

    # test methods

    def test_add_dataset_arg_single_output(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.OUTPUT
        exp_pos = ('--{}'.format(self.out_alss['dataset']), )
        self.dset_args.add_dataset_arg_single(mock_parser, atype,
                                              input_metavars=self.i_meta,
                                              output_metavars=self.o_meta)
        self._assert_call_single(mock_parser, exp_pos, 'store', atype, False)

    def test_add_dataset_arg_single_short_alias(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.OUTPUT
        exp_pos = ('-d', '--{}'.format(self.out_alss['dataset']))
        self.dset_args.add_dataset_arg_single(mock_parser, atype,
                                              short_alias='d',
                                              input_metavars=self.i_meta,
                                              output_metavars=self.o_meta)
        self._assert_call_single(mock_parser, exp_pos, 'store', atype, False)

    def test_add_dataset_arg_single_input(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.INPUT
        exp_pos = ('--{}'.format(self.in_alss['dataset']), )
        self.dset_args.add_dataset_arg_single(mock_parser, atype,
                                              input_metavars=self.i_meta,
                                              output_metavars=self.o_meta)
        self._assert_call_single(mock_parser, exp_pos, 'store', atype, False)

    def test_add_dataset_arg_single_multiple(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.OUTPUT
        exp_pos = ('--{}'.format(self.out_alss['dataset']), )
        self.dset_args.add_dataset_arg_single(mock_parser, atype,
                                              input_metavars=self.i_meta,
                                              output_metavars=self.o_meta,
                                              multiple=True)
        self._assert_call_single(mock_parser, exp_pos, 'append', atype, True)

    def test_add_dataset_arg_double_output(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.OUTPUT
        exp_name_pos = ('--{}'.format(self.out_alss['dataset name']), )
        exp_dir_pos = ('--{}'.format(self.out_alss['dataset directory']), )
        self.dset_args.add_dataset_arg_double(mock_parser, atype)
        self._assert_call_double(mock_parser, exp_name_pos, exp_dir_pos, atype)

    def test_add_dataset_arg_double_name_alias(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.OUTPUT
        exp_name_pos = ('-n', '--{}'.format(self.out_alss['dataset name']))
        exp_dir_pos = ('--{}'.format(self.out_alss['dataset directory']), )
        self.dset_args.add_dataset_arg_double(mock_parser, atype,
                                              name_short_alias='n')
        self._assert_call_double(mock_parser, exp_name_pos, exp_dir_pos, atype)

    def test_add_dataset_arg_double_dir_alias(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.OUTPUT
        exp_name_pos = ('--{}'.format(self.out_alss['dataset name']), )
        exp_dir_pos = ('-d', '--{}'.format(self.out_alss['dataset directory']))
        self.dset_args.add_dataset_arg_double(mock_parser, atype,
                                              dir_short_alias='d')
        self._assert_call_double(mock_parser, exp_name_pos, exp_dir_pos, atype)

    def test_add_dataset_arg_double_input(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.INPUT
        exp_name_pos = ('--{}'.format(self.in_alss['dataset name']), )
        exp_dir_pos = ('--{}'.format(self.in_alss['dataset directory']), )
        self.dset_args.add_dataset_arg_double(mock_parser, atype)
        self._assert_call_double(mock_parser, exp_name_pos, exp_dir_pos, atype)

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
        all_types = ds.ALL_ITEM_TYPES
        self.required = {all_types[idx]: (True if idx % 2 == 0 else False)
                         for idx in range(len(all_types))}
        self.item_args = cargs.item_types_args(in_item_prefix=self.in_prefix,
                                               out_item_prefix=self.out_prefix)

    # test methods

    def test_add_item_type_args_input(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.INPUT
        self.item_args.add_item_type_args(mock_parser, atype, self.required)
        self.assertEqual(mock_parser.add_argument.call_count,
                         len(ds.ALL_ITEM_TYPES))
        for idx in range(len(ds.ALL_ITEM_TYPES)):
            item_type = ds.ALL_ITEM_TYPES[idx]
            exp_pos = ('--{}_{}'.format(self.in_prefix, item_type), )
            act_pos = mock_parser.add_argument.call_args_list[idx][0]
            self.assertEqual(exp_pos, act_pos)
            kw = mock_parser.add_argument.call_args_list[idx][1]
            self.assertEqual(kw['required'], self.required[item_type])

    def test_add_item_type_args_output(self):
        mock_parser = mock.MagicMock()
        atype = cargs.arg_type.OUTPUT
        self.item_args.add_item_type_args(mock_parser, atype, self.required)
        self.assertEqual(mock_parser.add_argument.call_count,
                         len(ds.ALL_ITEM_TYPES))
        for idx in range(len(ds.ALL_ITEM_TYPES)):
            item_type = ds.ALL_ITEM_TYPES[idx]
            exp_pos = ('--{}_{}'.format(self.out_prefix, item_type), )
            act_pos = mock_parser.add_argument.call_args_list[idx][0]
            self.assertEqual(exp_pos, act_pos)
            kw = mock_parser.add_argument.call_args_list[idx][1]
            self.assertEqual(kw['required'], self.required[item_type])

    def test_get_item_type_args_input(self):
        attr_names = ['{}_{}'.format(self.in_prefix, item_type)
                      for item_type in ds.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(ds.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        vals = self.item_args.get_item_types(args, cargs.arg_type.INPUT)
        self.assertDictEqual(dict(zip(ds.ALL_ITEM_TYPES, attr_vals)), vals)

    def test_get_item_type_args_output(self):
        attr_names = ['{}_{}'.format(self.out_prefix, item_type)
                      for item_type in ds.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(ds.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        vals = self.item_args.get_item_types(args, cargs.arg_type.OUTPUT)
        self.assertDictEqual(dict(zip(ds.ALL_ITEM_TYPES, attr_vals)), vals)

    def test_check_item_type_args_input_no_exception(self):
        attr_names = ['{}_{}'.format(self.in_prefix, item_type)
                      for item_type in ds.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(ds.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        self.item_args.check_item_type_args(args, cargs.arg_type.INPUT)

    def test_check_item_type_args_output_no_exception(self):
        attr_names = ['{}_{}'.format(self.out_prefix, item_type)
                      for item_type in ds.ALL_ITEM_TYPES]
        attr_vals = [True if idx % 3 == 0 else False
                     for idx in range(len(ds.ALL_ITEM_TYPES))]
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        self.item_args.check_item_type_args(args, cargs.arg_type.OUTPUT)

    def test_check_item_type_args_input_exception(self):
        attr_names = ['{}_{}'.format(self.in_prefix, item_type)
                      for item_type in ds.ALL_ITEM_TYPES]
        attr_vals = [False] * len(ds.ALL_ITEM_TYPES)
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        try:
            self.item_args.check_item_type_args(args, cargs.arg_type.INPUT)
            self.fail('Item type checking did not throw an error')
        except Exception:
            pass

    def test_check_item_type_args_output_exception(self):
        attr_names = ['{}_{}'.format(self.out_prefix, item_type)
                      for item_type in ds.ALL_ITEM_TYPES]
        attr_vals = [False] * len(ds.ALL_ITEM_TYPES)
        args = coll.namedtuple('args', attr_names)(*attr_vals)
        try:
            self.item_args.check_item_type_args(args, cargs.arg_type.OUTPUT)
            self.fail('Item type checking did not throw an error')
        except Exception:
            pass


class TestModuleFunctions(unittest.TestCase):

    # helper methods (custom assert)

    def _assert_call_num_range(self, mock_parser, arg_name, s_alias=None,
                               desc=None):
        if s_alias is not None:
            pos = ('-{}'.format(s_alias), '--{}'.format(arg_name))
        else:
            pos = ('--{}'.format(arg_name), )
        if desc is not None:
            help = ('{}. MIN == MAX implies a constant value.'
                    .format(desc))
        else:
            help = ('Range of {} values. MIN == MAX implies a constant value.'
                    .format(arg_name))
        self.assertEqual(mock_parser.add_argument.call_count, 1)
        self.assertEqual(mock_parser.add_argument.call_args[0], pos)
        kw = mock_parser.add_argument.call_args[1]
        self.assertEqual(kw['help'], help)

    # test methods

    def test_add_number_range_arg_default(self):
        mock_parser = mock.MagicMock()
        cargs.add_number_range_arg(mock_parser, 'foo')
        self._assert_call_num_range(mock_parser, 'foo')

    def test_add_number_range_arg_short_alias(self):
        mock_parser = mock.MagicMock()
        cargs.add_number_range_arg(mock_parser, 'foo', short_alias='f')
        self._assert_call_num_range(mock_parser, 'foo', s_alias='f')

    def test_add_number_range_arg_custom_desc(self):
        mock_parser = mock.MagicMock()
        desc = 'Some arguemnt'
        cargs.add_number_range_arg(mock_parser, 'foo', arg_desc=desc)
        self._assert_call_num_range(mock_parser, 'foo', desc=desc)


if __name__ == '__main__':
    unittest.main()
