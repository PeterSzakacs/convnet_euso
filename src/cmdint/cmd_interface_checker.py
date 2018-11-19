import os
import argparse

import cmdint.common_args as common_args

class cmd_interface():

    def __init__(self):
        self.default_logdir = os.path.join('/run/user/', str(os.getuid()), 'convnet_checker/')
        self.parser = argparse.ArgumentParser(description="Evaluate trained network(s) with data")
        common_args.add_input_type_dataset_args(self.parser)

        # neural network and trained model parameters
        self.parser.add_argument('-n', '--networks', required=True, nargs=2, action='append', metavar=('NETWORK_NAME', 'MODEL_FILE'),
                                help='names of network modules to use and a respective model file containing a trained model.')
        # different input specification methods
        ## dataset files identified by name, also all located in the same directory
        self.parser.add_argument('--name', required=True,
                                help=('name of the dataset, input, targets and meta file names are constructed from these.'))
        self.parser.add_argument('--srcdir', required=True,
                                help=('directory containing input data and target files.'))
        ## single acquisition file in CERN ROOT format with optional L1 trigger file
        # self.parser.add_argument('--acqfile',
        #                          help=('the acquisition ROOT file storing air shower data. Can be specified alongside --triggerfile.'))
        # self.parser.add_argument('--triggerfile',
        #                          help=('the L1 triggers file for the given acquisition file (--acqfile).'))

        # How many frames (at most) to use:
        self.parser.add_argument('--eval_numframes', type=int,
                                help=('number of frames out of the passed dataset to use for evaluation (Selects first n frames).'))

        # misc
        self.parser.add_argument('--logdir', default=self.default_logdir,
                                help=('Directory to store output logs, default: "/run/user/$USERID/convnet_checker/".'
                                      ' If a non-default directory is used, it must exist prior to calling this script,'
                                      ' otherwise an error will be thrown.'))
        self.parser.add_argument('--tablesize', type=int,
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

        common_args.check_input_type_dataset_args(args)
        args.item_types = common_args.input_type_dataset_args_to_dict(args)

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
        if args.logdir != self.default_logdir and not os.path.exists(args.logdir):
            raise ValueError('Non-default logging output directory {} does not exist'.format(args.logdir))

        return args
