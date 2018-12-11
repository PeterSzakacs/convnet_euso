import os
import argparse

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=('Merge multiple datasets into a single dataset'
                                                           ' by concatenating them in the order they are passed'))
        self.parser.add_argument('-d', '--datasets', required=True, nargs=2, action='append', metavar=('NAME', 'SRCDIR'),
                                help='input datasets to merge together specified via name and source directory')
        self.parser.add_argument('-n', '--name', required=True,
                                help='name for the new dataset')
        self.parser.add_argument('-o', '--outdir', required=True, default=os.path.curdir,
                                help='directory to store output dataset files, default: current directory')
        self.parser.add_argument('--delete_original', action='store_true',
                                help='delete the original input datasets')


    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        if not os.path.isdir(args.outdir):
            raise ValueError("Invalid output directory {}".format(args.outdir))
        if len(args.datasets) < 2:
            raise ValueError("At least 2 datasets must be passed")

        return args
