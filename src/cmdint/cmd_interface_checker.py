import os
import sys
import argparse

import cmdint.common.argparse_types as atypes
import cmdint.common.args as cargs
import cmdint.common.dataset_args as dargs
import cmdint.common.network_args as net_args

class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Evaluate trained network model with given dataset")
        parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                            default=sys.stdout,
                            help=('name of output TSV to write to. If not '
                                  'provided, output to stdout.'))

        # dataset input
        atype = dargs.arg_type.INPUT
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        dset_args = dargs.DatasetArgs(input_aliases=in_aliases)
        item_args = dargs.ItemTypeArgs()
        group = parser.add_argument_group(title="Input dataset")
        dset_args.add_dataset_arg_double(group, atype)
        item_args.add_item_type_args(group, atype)
        # slice of dataset items to use for evaluation
        group.add_argument('--start_item', default=0, type=int,
                           help=('index of first dataset item to use for '
                                 'evaluation.'))
        group.add_argument('--stop_item', default=None, type=int,
                           help=('index of the dataset item after the last '
                                 'item to use for evaluation.'))

        # trained neural network model
        group = parser.add_argument_group('Neural network settings')
        net_args.add_network_arg(group, short_alias='n')
        net_args.add_model_file_arg(group, short_alias='m', required=True)

        # misc
        parser.add_argument('--usecpu', action='store_true',
                            help=('Use host CPU instead of the CUDA device. '
                                  'On systems without a dedicated CUDA device '
                                  'and no CUDA-enabled version  of tensorflow '
                                  'installed, this flag has no effect.'))
        # parser.add_argument('--onlyerr', action='store_true',
        #                     help=('Include only failed predictions in the '
        #                           'output'))

        # metafields order of the generated TSV
        meta_args = cargs.MetafieldOrderArg()
        g_title = "Order of metadata fields in the generated TSV"
        meta_args.add_metafields_order_arg(parser, group_title=g_title)

        self.parser = parser
        self.meta_args = meta_args
        self.dset_args = dset_args
        self.item_args = item_args


    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = dargs.arg_type.INPUT
        args.item_types = self.item_args.get_item_types(args, atype)
        args.meta_order = self.meta_args.get_metafields_order(args)

        network_name, model_file = args.network, args.model_file
        if not os.path.exists(model_file + ".meta"):
            raise ValueError('Model file {} for network {} does not exist: '
                             .format(model_file, network_name))

        return args
