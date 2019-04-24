import os
import sys
import argparse

import cmdint.common.argparse_types as atypes
import cmdint.common.args as cargs
import cmdint.common.dataset_args as dargs


class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description=('Create classification report in HTML format from '
                         'provided evaluation results in TSV format'))

        # input tsv
        parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                            default=sys.stdin,
                            help=('name of input TSV to read from. If not '
                                  'provided, read from stdin.'))

        # output settings
        group = parser.add_argument_group(title="Output settings")
        group.add_argument('--tablesize', type=atypes.int_range(1),
                           help=('Maximum number of table rows per html '
                                 'report file.'))
        group.add_argument('--logdir',
                           help=('Directory to store output logs. If a '
                                 'non-default directory is used, it must '
                                 'exist prior to calling this script.'))
        item_args = dargs.ItemTypeArgs(out_item_prefix='add')
        help = {k: 'add image placeholder for {}'.format(desc)
                for k, desc in item_args.item_descriptions.items()}
        item_args.add_item_type_args(group, dargs.arg_type.OUTPUT, help=help)

        # meta-information to include in report headers
        g_title = "Meta-information to include in report headers"
        group = parser.add_argument_group(title=g_title)
        group.add_argument('-n', '--network', required=True, nargs=2,
                           metavar=('NETWORK_NAME', 'MODEL_FILE'),
                           help=('name of network module used and '
                                 'corresponding trained model file.'))
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        dset_args = dargs.DatasetArgs(input_aliases=in_aliases)
        dset_args.add_dataset_arg_double(group, dargs.arg_type.INPUT,
                                         required=False)

        # order of metadata columns in the report
        g_title = 'Metadata column order of report files'
        meta_args = cargs.MetafieldOrderArg()
        meta_args.add_metafields_order_arg(parser, group_title=g_title)
        self.parser = parser
        self.dset_args = dset_args
        self.item_args = item_args
        self.meta_args = meta_args

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = dargs.arg_type.OUTPUT
        args.item_types = self.item_args.get_item_types(args, atype)

        atype = dargs.arg_type.INPUT
        args.name, args.srcdir = self.dset_args.get_dataset_double(args, atype)

        args.meta_order = self.meta_args.get_metafields_order(args)

        return args
