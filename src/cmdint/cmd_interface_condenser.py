import os
import argparse

import utils.common_utils as cutils
import cmdint.common_args as common_args

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Use trained model to evaluate recorded flight data")
        self.parser.add_argument('-f', '--filelist', required=True,
                                help=('input files list in TSV format. Mutually exclusive with --acqfile,'
                                      ' either use this or that'))
        self.parser.add_argument('-n', '--name', required=True,
                                help=('name of the output dataset, used as part of the filename for all files'))
        self.parser.add_argument('-o', '--outdir', required=True, default=os.path.curdir,
                                help=('directory to store output dataset files'))
        input_type = self.parser.add_mutually_exclusive_group(required=True)
        input_type.add_argument('--simu', action='store_true', 
                                help=('apply simu transformer when creating dataset items'))
        input_type.add_argument('--flight', action='store_true', 
                                help=('apply flight transformer when creating dataset items'))
        input_type.add_argument('--custom', metavar=['TARGET', 'START_GTU', 'END_GTU'], nargs='+',
                                action=common_args.allowed_lengths(lengths=[1,3]),
                                help=('apply custom transformer when creating dataset items. Accepts range of'
                                      ' frames and static target (noise or shower) to apply when extracting all'
                                      ' events in the passed filelist.'))

        common_args.add_packet_args(self.parser)
        common_args.add_output_type_dataset_args(self.parser)


    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        if not os.path.exists(args.outdir):
            raise ValueError("Output directory {} does not exist".format(args.outdir))
        if not os.path.isdir(args.outdir):
            raise ValueError("Output directory {} is not a directory".format(args.outdir))
        if not os.path.exists(args.filelist):
            raise ValueError("List of files to process {} does not exist".format(args.filelist))

        args.template = common_args.packet_args_to_packet_template(args)
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

        common_args.check_output_type_dataset_args(args)
        args.item_types = common_args.output_type_dataset_args_to_dict(args)

        return args
