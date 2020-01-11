import os
import argparse
import textwrap

import cmdint.common.argparse_types as atypes
import cmdint.common.args as cargs
import cmdint.common.dataset_args as dargs
import dataset.constants as cons


_TARGET_ARG_HELP = textwrap.dedent(f'''\
        target assignment method to use for all extracted items along with 
        relevant arguments for it. Supported methods and their arguments:
        --target STATIC ({"|".join(cons.CLASSIFICATION_TARGETS)})
        set chosen static value for all items
        --target BIN_COLUMN <COLUMN_NAME>
        set value from metadata column (must be included in extra_metafields)
        ''')


class CmdInterface:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Create dataset from multiple files with packets",
            formatter_class=argparse.RawTextHelpFormatter
        )

        # global settings
        parser.add_argument('--max_cache_size', default=40,
                            type=atypes.int_range(1),
                            help=('maximum size of parsed files cache'))
        parser.add_argument('--num_evicted', default=10,
                            type=atypes.int_range(1),
                            help=('number of cache entires to evict when the '
                                  'cache gets full'))

        # input settings
        group = parser.add_argument_group(title='Input settings')
        packet_args = cargs.PacketArgs()
        packet_args.add_packet_arg(group)
        group.add_argument('-f', '--filelist', required=True,
                           help=('input files list in TSV format'))

        # output (dataset) settings
        group = parser.add_argument_group(title='Output settings')
        out_aliases = {'dataset name': 'name', 'dataset directory': 'outdir'}
        dset_args = dargs.DatasetArgs(output_aliases=out_aliases)
        dset_args.add_dataset_arg_double(group, dargs.arg_type.OUTPUT,
                                         dir_short_alias='d', dir_default='.',
                                         name_short_alias='n')

        # output (dataset) data item settings
        group = parser.add_argument_group(title='Data item settings')
        item_args = dargs.ItemTypeArgs()
        item_args.add_item_type_args(group, dargs.arg_type.OUTPUT)
        group.add_argument('--dtype', default='float32',
                           help='cast extracted items to the given numpy data '
                                'type (default: %(default)s))')

        # output (dataset) target settings
        group = parser.add_argument_group(title='Item target settings')
        group.add_argument('--target', required=True, nargs=2,
                           metavar=('METHOD', 'ARGS'),
                           help=_TARGET_ARG_HELP)

        # output (dataset) metadata settings
        group = parser.add_argument_group(title='Metadata settings')
        group.add_argument('--extra_metafields', nargs='+', default=[],
                           metavar='FIELD',
                           help=('additional fields in the event list to '
                                 'include in dataset metadata'))

        subparsers = parser.add_subparsers(dest="converter",
            help='Packet to item conversion methods')

        def_m = subparsers.add_parser("default",
                                      help=("Convert events to dataset items "
                                            "using default transformer"))
        def_m.add_argument('--gtu_range', type=atypes.int_range(0), nargs=2,
                           metavar=('START_GTU', 'STOP_GTU'), required=True,
                           help=('range of GTUs to use'))
        def_m.add_argument('--packet_idx', type=atypes.int_range(0),
                          required=True,
                          help=('index of packet to use'))

        apack = subparsers.add_parser("allpack",
                                      help=("Convert events to dataset items "
                                            "using all_packets transformer"))
        apack.add_argument('--gtu_range', type=atypes.int_range(0), nargs=2,
                           metavar=('START_GTU', 'STOP_GTU'), required=True,
                           help=('range of GTUs containing shower.'))

        gpack = subparsers.add_parser("gtupack",
                                      help=("Convert events to dataset items "
                                            "using gtu_in_packet transformer"))
        gpack.add_argument('--num_gtu_around', type=atypes.int_range(0),
                           nargs=2, metavar=('NUM_BEFORE', 'NUM_AFTER'),
                           help=('number of GTU/frames before and after '
                                 'gtu_in_packet to include in dataset items'))
        gpack.add_argument('--no_bounds_adjust', action='store_true',
                           help=('do not shift the frames window if part of '
                                 'it is out of packet bounds. An exception '
                                 'will be raised instead '))

        self.parser = parser
        self.packet_args = packet_args
        self.dset_args = dset_args
        self.item_args = item_args

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        if not os.path.isfile(args.filelist):
            raise ValueError("Invalid filelist {}".format(args.filelist))
        if not os.path.isdir(args.outdir):
            raise ValueError("Invalid output directory {}".format(args.outdir))

        args.template = self.packet_args.packet_arg_to_template(args)

        atype = dargs.arg_type.OUTPUT
        args.item_types = self.item_args.get_item_types(args, atype)

        self._parse_converter_arg(args)
        self._parse_target_arg(args)

        return args

    @staticmethod
    def _parse_target_arg(args):
        raw_value = args.target
        method = raw_value[0].upper()
        if method == 'STATIC':
            handler_args = {
                'target_value': cons.CLASSIFICATION_TARGETS[raw_value[1]]
            }
        elif method == 'BIN_COLUMN':
            handler_args = {
                'column_name': raw_value[1]
            }
        else:
            raise ValueError(f'Unknown target assignment method {method}')
        args.target_handler_type = method
        args.target_handler_args = handler_args

    @staticmethod
    def _parse_converter_arg(args):
        converter = args.converter
        if converter == 'gtupack':
            _before, _after = args.num_gtu_around[0:2]
            transformer_args = {
                "num_gtu_before": _before, "num_gtu_after": _after,
                "adjust_if_out_of_bounds": not args.no_bounds_adjust,
            }
        elif converter == 'allpack':
            _start, _stop = args.gtu_range[0:2]
            transformer_args = {
                "start_gtu": _start, "stop_gtu": _stop,
            }
        elif converter == 'default':
            _packet_id, (_start, _stop) = args.packet_idx, args.gtu_range
            transformer_args = {
                "start_gtu": _start, "stop_gtu": _stop,
                "packet_id": _packet_id,
            }
        else:
            # in case later we add another converter and subparser
            raise ValueError(f"Unknown converter {converter}")
        args.event_transformer = converter
        args.event_transformer_args = transformer_args
