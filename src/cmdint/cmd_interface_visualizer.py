import os
import argparse

import cmdint.argparse_types as atypes
import cmdint.common_args as cargs

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Visualize dataset items")

        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        self.dset_args = cargs.dataset_args(input_aliases=in_aliases)
        self.item_args = cargs.item_types_args()
        self.dset_args.add_dataset_arg_double(self.parser,
                                              cargs.arg_type.INPUT,
                                              required=True,
                                              dir_default=os.path.curdir)
        self.item_args.add_item_type_args(self.parser, cargs.arg_type.INPUT)

        self.parser.add_argument('--outdir', default=os.path.curdir,
                                help=('directory to store output logs, default: current directory. If a non-default'
                                      ' directory is used, it must exist prior to calling this script, otherwise an'
                                      ' error will be thrown'))
        self.parser.add_argument('--start_item', default=0, type=int,
                                help=('index of first item to visualize.'))
        self.parser.add_argument('--stop_item', default=None, type=int,
                                help=('index of the item after the last item '
                                      'to visualize.'))
        dataset_type = self.parser.add_mutually_exclusive_group(required=True)
        dataset_type.add_argument('--simu', action='store_true', help=('dataset created from a multitude of source npy files with simulated data'))
        dataset_type.add_argument('--synth', action='store_true', help=('dataset created using the data_generator script'))
        dataset_type.add_argument('--flight', action='store_true', help=('dataset created from recorded flight data in CERN ROOT format'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        if not os.path.isdir(args.outdir):
            raise ValueError("Invalid output directory {}".format(args.outdir))

        atype = cargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)

        return args