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
        self.meta_args = cargs.metafield_order_arg()

        # input tsv
        self.parser.add_argument('infile', nargs='?',
                                 type=argparse.FileType('r'),
                                 default=sys.stdin,
                                 help=('name of input TSV to read from. If '
                                       'not provided, read from stdin.'))

        # trained neural network model
        self.parser.add_argument('-n', '--network', required=True, nargs=2,
                                 metavar=('NETWORK_NAME', 'MODEL_FILE'),
                                 help=('name of network module used and '
                                       'corresponding trained model file.'))
        self.parser.add_argument('--tablesize', type=atypes.int_range(1),
                                 help=('Maximum number of table rows per html '
                                        'report file.'))
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

        # order of metadata columns in the report
        # dataset_type = self.parser.add_mutually_exclusive_group(required=True)
        self.meta_args.add_metafields_order_arg(
            self.parser, group_title='Metadata column order of report files')

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = cargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)

        args.meta_order = self.meta_args.get_metafields_order(args)
        args.name, args.srcdir = self.dset_args.get_dataset_double(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)

        return args
