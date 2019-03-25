import os
import argparse

import cmdint.common.argparse_types as atypes
import cmdint.common.dataset_args as dargs

class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(description="Visualize dataset items")

        # input dataset settings
        group = parser.add_argument_group(title="Input dataset")
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        dset_args = dargs.dataset_args(input_aliases=in_aliases)
        dset_args.add_dataset_arg_double(group, dargs.arg_type.INPUT,
                                         required=True,
                                         dir_default=os.path.curdir)
        item_args = dargs.item_types_args()
        item_args.add_item_type_args(group, dargs.arg_type.INPUT)
        group.add_argument('--start_item', default=0, type=int,
                           help=('index of first item to visualize.'))
        group.add_argument('--stop_item', default=None, type=int,
                           help=('index of the item after the last item to '
                                 'visualize.'))

        # output settings
        group = parser.add_argument_group(title="Output settings")
        group.add_argument('--outdir', default=os.path.curdir,
                           help=('directory to store output images. If a '
                                 'non-default directory is used, it must '
                                 'exist prior to calling this script. '
                                 'Default: current directory. Images '
                                 'are stored under outdir/img/<item_type>'))
        group.add_argument('-f', '--force_overwrite', action='store_true',
                           help=('overwrite any existing items under outdir '
                                 'having the same name as generated items'))

        # metadat to text converter
        group = parser.add_argument_group(title="Metadata to text converter")
        m_conv = group.add_mutually_exclusive_group(required=True)
        m_conv.add_argument('--simu', action='store_const', const='simu',
                            help=('Simu metadata converter'))
        m_conv.add_argument('--synth', action='store_const', const='synth',
                            help=('Synth metadata converter'))
        m_conv.add_argument('--flight', action='store_const', const='flight',
                            help=('Flight metadata converter'))

        self.parser = parser
        self.dset_args = dset_args
        self.item_args = item_args

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        if not os.path.isdir(args.outdir):
            raise ValueError("Invalid output directory {}".format(args.outdir))

        atype = dargs.arg_type.INPUT
        args.item_types = self.item_args.get_item_types(args, atype)

        args.meta_to_text_conv = args.simu or args.flight or args.synth

        return args