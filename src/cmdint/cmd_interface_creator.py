import os
import argparse

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
        input_type.add_argument('--simu', action='store_true', help=('specifies to use simulated air shower data in npy format'))
        input_type.add_argument('--flight', action='store_true', help=('specifies to use recorded flight data in CERN ROOT format'))

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

        common_args.check_output_type_dataset_args(args)
        args.item_types = common_args.output_type_dataset_args_to_dict(args)

        return args
