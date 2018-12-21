import os
import argparse

import utils.common_utils as cutils
import cmdint.argparse_types as atypes
import cmdint.common_args as cargs
import cmdint.arparse_actions as actions

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Create single dataset from a list of files")
        self.packet_args = cargs.packet_args()
        self.item_args = cargs.item_types_args()
        self.dset_args = cargs.dataset_args(output_aliases={
            'dataset name': 'name', 'dataset directory': 'outdir'})

        self.parser.add_argument('-f', '--filelist', required=True,
                                help=('input files list in TSV format. Mutually exclusive with --acqfile,'
                                      ' either use this or that'))
        self.packet_args.add_packet_arg(self.parser)
        self.dset_args.add_dataset_arg_double(self.parser,
                                              cargs.arg_type.OUTPUT,
                                              name_short_alias='n',
                                              dir_short_alias='d',
                                              dir_default=os.path.curdir)
        self.item_args.add_item_type_args(self.parser, cargs.arg_type.OUTPUT)
        self.parser.add_argument('--max_cache_size', default=40, 
                                type=atypes.int_range(1),
                                help=('maximum size of parsed files cache'))
        input_type = self.parser.add_mutually_exclusive_group(required=True)
        input_type.add_argument('--simu', action='store_true',
                                help=('apply simu transformer when creating dataset items'))
        input_type.add_argument('--flight', action='store_true',
                                help=('apply flight transformer when creating dataset items'))
        input_type.add_argument('--custom', metavar=['TARGET', 'START_GTU', 'END_GTU'], nargs='+',
                                action=actions.allowed_lengths(lengths=[1,3]),
                                help=('apply custom transformer when creating dataset items. Accepts range of'
                                      ' frames and static target (noise or shower) to apply when extracting all'
                                      ' events in the passed filelist.'))


    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        if not os.path.isfile(args.filelist):
            raise ValueError("Invalid filelist {}".format(args.filelist))
        if not os.path.isdir(args.outdir):
            raise ValueError("Invalid output directory {}".format(args.outdir))

        args.template = self.packet_args.packet_arg_to_template(args)
        if not (args.simu or args.flight):
            args.target = args.custom[0]
            if args.target.lower() == 'shower':
                args.target = [0, 1]
            else:
                args.target = [1, 0]
            gtus = args.custom[1:3] if len(args.custom) == 3 else [None, None]
            if gtus != [None, None]:
                try:
                    args.start_gtu, args.end_gtu = int(gtus[0]), int(gtus[1])
                except ValueError:
                    raise TypeError('Not a valid frame range in custom transformer: {}'.format(args.custom[1:3]))
            else:
                args.start_gtu, args.end_gtu = gtus[0:2]

        atype = cargs.arg_type.OUTPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)

        return args
