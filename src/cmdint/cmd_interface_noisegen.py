import argparse

import cmdint.common.argparse_types as atypes
import cmdint.common.args as cargs
import cmdint.common.dataset_args as dargs

class CmdInterface():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Create dataset containing simulated noise")
        out_aliases = {'dataset name': 'name', 'dataset directory': 'outdir'}
        packet_args = cargs.PacketArgs()
        dset_args = dargs.dataset_args(output_aliases=out_aliases)
        item_args = dargs.item_types_args()

        group = parser.add_argument_group("Noise settings")
        cargs.add_number_range_arg(group, 'bg_lambda', required=True,
                                   arg_type=atypes.float_range(0),
                                   arg_desc=('Pixel values average (Poisson '
                                             'distribution lambda)'))
        group.add_argument('--precision', type=atypes.int_range(1), default=4,
                           help=('Number of decimal digits to round generated '
                                 'bg_lambda values to'))
        group.add_argument('--seed',
                           help=('Random seed to use for generating bg_lambda '
                                 'values'))

        group = parser.add_argument_group('Output dataset settings')
        # packet dimensions
        packet_args.add_packet_arg(group, required=True)
        # output dataset
        atype = dargs.arg_type.OUTPUT
        dset_args.add_dataset_arg_double(group, atype)
        item_args.add_item_type_args(group, atype)
        group.add_argument('--num_items', required=True,
                           type=atypes.int_range(1),
                           help='Number of dataset items to generate')
        group.add_argument('--dtype', default='uint8',
                           help='Data type of dataset items (default: uint8)')
        self.parser = parser
        self.packet_args = packet_args
        self.dset_args = dset_args
        self.item_args = item_args

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)
        args_dict = {}

        packet_templ = self.packet_args.packet_arg_to_template(args)
        args_dict['packet_shape'] = packet_templ.packet_shape

        atype = dargs.arg_type.OUTPUT
        name, outdir = self.dset_args.get_dataset_double(args, atype)
        args_dict['name'], args_dict['outdir'] = name, outdir
        args_dict['item_types'] = self.item_args.get_item_types(args, atype)
        args_dict['num_items'], args_dict['dtype'] = args.num_items, args.dtype

        args_dict['bg_lambda'] = args.bg_lambda
        args_dict['seed'], args_dict['precision'] = args.seed, args.precision

        return args_dict
