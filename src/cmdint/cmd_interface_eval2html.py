import os
import sys
import argparse

import cmdint.argparse_types as atypes
import cmdint.common_args as cargs


class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=('Randomly shuffle dataset a given number of times'))
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        self.dset_args = cargs.dataset_args(input_aliases=in_aliases)
        self.item_args = cargs.item_types_args()

        # input tsv
        self.parser.add_argument('infile', nargs='?',
                                 type=argparse.FileType('r'),
                                 default=sys.stdin,
                                 help=('name of input TSV to read from. If '
                                       'not provided, read from stdin.'))

        # trained neural network model
        self.parser.add_argument('-n', '--network', required=True, nargs=2,
                                 metavar=('NETWORK_NAME', 'MODEL_FILE'),
                                 help=('name of network module to use and a '
                                       'corresponding trained model file.'))
        self.parser.add_argument('--tablesize', type=atypes.int_range(1),
                        help=('Maximum number of table rows for every report '
                              'file.'))
        self.parser.add_argument('--logdir',
                                help=('Directory to store output logs. If a '
                                      'non-default directory is used, it must '
                                      'exist prior to calling this script.'))

        # evaluation dataset name and directory, optional information in report
        # headers
        atype = cargs.arg_type.INPUT
        self.dset_args.add_dataset_arg_double(self.parser, atype,
                                              required=False)
        self.item_args.add_item_type_args(self.parser, atype)

        # metadata present in the input TSV records
        dataset_type = self.parser.add_mutually_exclusive_group(required=True)
        dataset_type.add_argument('--simu', action='store_true',
                                  help=('dataset created from multiple source '
                                        'npy files with simulated data'))
        dataset_type.add_argument('--synth', action='store_true',
                                  help=('dataset created using data_generator '
                                        'script'))
        dataset_type.add_argument('--flight', action='store_true',
                                  help=('dataset created from recorded flight '
                                        'data in CERN ROOT format'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = cargs.arg_type.INPUT
        args.name, args.srcdir = self.dset_args.get_dataset_double(args, atype)
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)

        return args
