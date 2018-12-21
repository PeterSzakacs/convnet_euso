import os
import argparse

import cmdint.argparse_types as atypes
import cmdint.common_args as cargs

class cmd_interface():

    def __init__(self):
        self.default_logdir = os.path.join('/run/user/', str(os.getuid()), 'convnet_checker/')
        self.parser = argparse.ArgumentParser(
            description="Evaluate trained network model(s) with given dataset")
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        self.dset_args = cargs.dataset_args(input_aliases=in_aliases)
        self.item_args = cargs.item_types_args()

        # dataset input
        atype = cargs.arg_type.INPUT
        self.dset_args.add_dataset_arg_double(self.parser, atype)
        self.item_args.add_item_type_args(self.parser, atype)

        # trained neural network model(s)
        self.parser.add_argument('-n', '--networks', required=True, nargs=2, action='append', metavar=('NETWORK_NAME', 'MODEL_FILE'),
                                help='names of network modules to use and a respective model file containing a trained model.')
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
        self.parser.add_argument('--logdir', default=self.default_logdir,
                                help=('Directory to store output logs, default: "/run/user/$USERID/convnet_checker/".'
                                      ' If a non-default directory is used, it must exist prior to calling this script,'
                                      ' otherwise an error will be thrown.'))
        self.parser.add_argument('--tablesize', type=atypes.int_range(1),
                                help=('Maximum number of table rows for every html report file.'))
        self.parser.add_argument('--usecpu', action='store_true',
                                help=('Use the CPU of the running machine instead of the CUDA device.'
                                      ' On systems without a dedicated CUDA device and no GPU version '
                                      ' of tensorflow installed, this flag has no effect.'))
        self.parser.add_argument('--onlyerr', action='store_true',
                                help=('Include only data about the failed predictions of the model in the generated report.'))
        dataset_type = self.parser.add_mutually_exclusive_group(required=True)
        dataset_type.add_argument('--simu', action='store_true', help=('dataset created from a multitude of source npy files with simulated data'))
        dataset_type.add_argument('--synth', action='store_true', help=('dataset created using the data_generator script'))
        dataset_type.add_argument('--flight', action='store_true', help=('dataset created from recorded flight data in CERN ROOT format'))


    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = cargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)

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
        # Check network modules and trained models built from those modules
        for network in args.networks:
            network_name = network[0]
            model_file = network[1]
            if not os.path.exists(model_file + ".meta"):
                raise ValueError('Model file {} for network {} does not exist: '.format(model_file, network_name))

        # Check logdir exists if not using default
        if args.logdir != self.default_logdir and not os.path.isdir(args.logdir):
            raise ValueError(('Invalid non-default logging directory {}'
                              .format(args.logdir)))

        return args
