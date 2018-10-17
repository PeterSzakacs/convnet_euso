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

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        common_args.check_input_type_dataset_args(args)
        args.item_types = common_args.input_type_dataset_args_to_dict(args)
        if not os.path.exists(args.outdir):
            raise ValueError('Non-default output directory does not exist')
        
        return args