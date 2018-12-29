import os
import argparse

import cmdint.common_args as cargs

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=('Merge multiple datasets into a single dataset by '
                         'concatenating them in the order they are passed'))
        in_aliases = {'dataset': 'datasets'}
        out_aliases = {'dataset name': 'name', 'dataset directory': 'outdir'}
        self.dset_args = cargs.dataset_args(input_aliases=in_aliases,
                                              output_aliases=out_aliases)
        self.dset_args.add_dataset_arg_single(self.parser,
                                              cargs.arg_type.INPUT,
                                              short_alias='d',
                                              multiple=True)
        self.dset_args.add_dataset_arg_double(self.parser,
                                              cargs.arg_type.OUTPUT,
                                              name_short_alias='n',
                                              dir_short_alias='o',
                                              dir_default='.')
        self.parser.add_argument('--delete_original', action='store_true',
                                help='delete the original input datasets')


    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        datasets = self.dset_args.get_dataset_single(args, 
                                                     cargs.arg_type.INPUT)
        if len(datasets) < 2:
            raise ValueError("At least 2 datasets must be passed")

        name, outdir = self.dset_args.get_dataset_double(args, 
                                                         cargs.arg_type.OUTPUT)
            if not os.path.isdir(outdir):
            raise ValueError("Invalid output directory {}".format(args.outdir))

        return args
