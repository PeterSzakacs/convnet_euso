import os
import sys
import argparse

import cmdint.argparse_types as atypes
import cmdint.common_args as cargs

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Evaluate trained network model(s) with given dataset")
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        self.dset_args = cargs.dataset_args(input_aliases=in_aliases)
        self.item_args = cargs.item_types_args()
        self.meta_args = cargs.metafield_order_arg()

        self.parser.add_argument('outfile', nargs='?',
                                 type=argparse.FileType('w'),
                                 default=sys.stdout,
                                 help=('name of output TSV to write to. If '
                                       'not provided, output to stdout.'))

        # dataset input
        atype = cargs.arg_type.INPUT
        self.dset_args.add_dataset_arg_double(self.parser, atype)
        self.item_args.add_item_type_args(self.parser, atype)

        # trained neural network model
        self.parser.add_argument('-n', '--network', required=True, nargs=2,
                                 metavar=('NETWORK_NAME', 'MODEL_FILE'),
                                 help=('name of network module to use and a '
                                       'corresponding trained model file.'))
        # different input specification methods
        ## single acquisition file in CERN ROOT format with optional L1 trigger file
        # self.parser.add_argument('--acqfile',
        #                          help=('the acquisition ROOT file storing air shower data. Can be specified alongside --triggerfile.'))
        # self.parser.add_argument('--triggerfile',
        #                          help=('the L1 triggers file for the given acquisition file (--acqfile).'))

        # How many frames (at most) to use:
        self.parser.add_argument('--eval_numframes', type=atypes.int_range(1),
                                help=('number of frames out of the passed dataset to use for evaluation (Selects first n frames).'))

        # misc
        self.parser.add_argument('--logdir',
                                help=('Directory to store output logs. If a '
                                      'non-default directory is used, it must '
                                      'exist prior to calling this script.'))
        self.parser.add_argument('--usecpu', action='store_true',
                                help=('Use the CPU of the running machine instead of the CUDA device.'
                                      ' On systems without a dedicated CUDA device and no GPU version '
                                      ' of tensorflow installed, this flag has no effect.'))
        self.parser.add_argument('--onlyerr', action='store_true',
                                help=('Include only failed predictions in the output'))

        # metafields order of the generated TSV
        self.meta_args.add_metafields_order_arg(
            self.parser,
            group_title="Order of metadata fields in the generated TSV")


    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = cargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)
        args.meta_order = self.meta_args.get_metafields_order(args)

        # != is basically XOR in python
        # acq_input = (args.acqfile != None)

        # valid inputs:
        #     name, srcdir
        #     acqfile, triggerfile

        # if npy_input == acq_input:
        #     raise ValueError(('Input method not specified or vague, please use just one of the following:'
        #                      ' srcdir and name, acqfile [and triggerfile]'))
        # Check if input method is valid
        # if npy_input:
            ## name/srcdir combination
            # args.npy, args.acq = True, False
        # else:
        #     ## single acqfile with optional triggerfile
        #     args.npy, args.acq = False, True
        #     if not os.path.exists(args.acqfile):
        #         raise ValueError('Input acquisition file "{}" does not exist'.format(args.acqfile))
        #     if args.triggerfile != None and not os.path.exists(args.triggerfile):
        #         raise ValueError('Input trigger file "{}" does not exist'.format(args.triggerfile))
        # Check network module and trained model
        network_name, model_file = args.network[0:2]
        if not os.path.exists(model_file + ".meta"):
            raise ValueError('Model file {} for network {} does not exist: '.format(model_file, network_name))

        # Check logdir exists if not using default
        # if args.logdir != self.default_logdir and not os.path.isdir(args.logdir):
        #     raise ValueError(('Invalid non-default logging directory {}'
        #                       .format(args.logdir)))

        return args
