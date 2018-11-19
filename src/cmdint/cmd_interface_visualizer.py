import os
import argparse

import cmdint.common_args as common_args

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Visualize dataset items")

        common_args.add_input_type_dataset_args(self.parser)

        self.parser.add_argument('--name', required=True,
                                help=('the name of the dataset'))
        self.parser.add_argument('--srcdir', required=True,
                                help=('directory containing dataset files'))
        self.parser.add_argument('--outdir', default=os.path.curdir,
                                help=('directory to store output logs, default: current directory. If a non-default'
                                      ' directory is used, it must exist prior to calling this script, otherwise an'
                                      ' error will be thrown'))
        self.parser.add_argument('--num_items', type=int,
                                help=('number of dataset items to visualize, default: all items'))
        dataset_type = self.parser.add_mutually_exclusive_group(required=True)
        dataset_type.add_argument('--simu', action='store_true', help=('dataset created from a multitude of source npy files with simulated data'))
        dataset_type.add_argument('--synth', action='store_true', help=('dataset created using the data_generator script'))
        dataset_type.add_argument('--flight', action='store_true', help=('dataset created from recorded flight data in CERN ROOT format'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        common_args.check_input_type_dataset_args(args)
        args.item_types = common_args.input_type_dataset_args_to_dict(args)
        if not os.path.exists(args.outdir):
            raise ValueError('Non-default output directory does not exist')

        return args