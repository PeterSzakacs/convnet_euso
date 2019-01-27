import os
import argparse

import utils.common_utils as cutils
import cmdint.argparse_types as atypes
import cmdint.common_args as cargs
import cmdint.arparse_actions as actions

class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Create dataset from multiple files with packets")
        parser.add_argument('--max_cache_size', default=40,
                            type=atypes.int_range(1),
                            help=('maximum size of parsed files cache'))

        group = parser.add_argument_group(title='Input settings')
        packet_args = cargs.packet_args()
        packet_args.add_packet_arg(group)
        group.add_argument('-f', '--filelist', required=True,
                           help=('input files list in TSV format'))

        group = parser.add_argument_group(title='Output settings')
        out_aliases = {'dataset name': 'name', 'dataset directory': 'outdir'}
        dset_args = cargs.dataset_args(output_aliases=out_aliases)
        dset_args.add_dataset_arg_double(group, cargs.arg_type.OUTPUT,
                                         dir_short_alias='d', dir_default='.',
                                         name_short_alias='n')
        item_args = cargs.item_types_args()
        item_args.add_item_type_args(group, cargs.arg_type.OUTPUT)
        group.add_argument('--target', required=True,
                           choices=('shower', 'noise'),
                           help=('classification target value to use for all '
                                 'items'))
        group.add_argument('--dtype', default='float32',
                           help='(numeric) datatype of all dataset items')

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

        atype = cargs.arg_type.OUTPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)

        return args
