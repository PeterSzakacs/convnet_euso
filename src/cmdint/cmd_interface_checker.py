import os
import argparse
from .base_cmd_interface import base_cmd_interface as bci

class cmd_interface(bci):

    def __init__(self):
        bci.__init__(self)
        self.default_logdir = os.path.join('/run/user/', str(os.getuid()), 'convnet_checker/')
        self.parser = argparse.ArgumentParser(description="Evaluate trained network(s) with data")

        # neural network and trained model parameters
        self.parser.add_argument('-n', '--networks', required=True, nargs=2, action='append', metavar=('NETWORK_NAME', 'MODEL_FILE'),
                                help='names of network modules to use and a respective model file containing a trained model.')
        # different input specification methods
        ## input files with frames and targets and optional TSV file containing metadata 
        self.parser.add_argument('-i', '--infile',
                                help=('the npy file storing simulated air shower data. If specified, implicitly overrides'
                                      ' --params, otherwise --params is used to construct the input file name. Must be'
                                      ' specified alongside --targetfile.'))
        self.parser.add_argument('-t', '--targetfile',
                                help=('the npy file storing expected neural network outputs (targets) for the data.' 
                                      ' If specified, implicitly overrides --params, otherwise --params is used'
                                      ' to construct the target file name. Must be specified alongside --infile.'))
        self.parser.add_argument('-m', '--metafile',
                                help=('optional TSV file storing dataset metadata.'))
        ## dataset files identified by name, also all located in the same directory
        self.parser.add_argument('--name',
                                help=('name of the dataset, input, targets and meta file names are constructed from these.'))
        self.parser.add_argument('--srcdir',
                                help=('directory containing input data and target files; only required if --name is used,'
                                      ' otherwise it is completely ignored.'))
        ## single acquisition file in CERN ROOT format with optional L1 trigger file
        self.parser.add_argument('--acqfile',
                                 help=('the acquisition ROOT file storing air shower data. Can be specified alongside --triggerfile.'))
        self.parser.add_argument('--triggerfile',
                                 help=('the L1 triggers file for the given acquisition file (--acqfile).'))
        
        # How many frames (at most) to use:
        self.parser.add_argument('--numframes', type=self._positive_int,
                                help=('number of frames out of the passed dataset to use for evaluation.'))

        # misc
        self.parser.add_argument('--logdir', default=self.default_logdir,
                                help=('Directory to store output logs, default: "/run/user/$USERID/convnet_checker/".'
                                      ' If a non-default directory is used, it must exist prior to calling this script,' 
                                      ' otherwise an error will be thrown.'))
        self.parser.add_argument('--tablesize', type=self._positive_int,
                                help=('Maximum number of table rows for every html report file.'))
        self.parser.add_argument('--usecpu', action='store_true',
                                help=('Use the CPU of the running machine instead of the CUDA device.'
                                      ' On systems without a dedicated CUDA device and no GPU version '
                                      ' of tensorflow installed, this flag has no effect.'))
        self.parser.add_argument('--noframes', action='store_true',
                                help=('Only create the actual report file but do not save images of frames in the dataset'
                                      ' Only do this if you are reevaluating on an existing dataset and those frames have'
                                      ' already been generated (as the report table will still reference them).'))
        self.parser.add_argument('--onlyerr', action='store_true',
                                help=('Include only data about the failed predictions of the model in the generated report.'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)
        
        use_srcdir = (args.srcdir != None)
        use_name = (args.name != None and use_srcdir)
        use_infiles = (args.infiles != None and args.targetfile != None)

        # != is basically XOR in python
        npy_input = (use_name != use_infiles)
        acq_input = (args.acqfile != None)

        # valid inputs:
        #     infile, triggerfile, [metafile]
        #     name, srcdir
        #     acqfile, triggerfile

        if npy_input == acq_input:
            raise ValueError(('Input method not specified or vague, please use just one of the following:'
                             ' srcdir and name, inputfiles and targetfile [and metafile] acqfile [and triggerfile]'))
        # Check if input method is valid
        if npy_input:
            ## name/srcdir and infile/targetfile/[metafile] combinations
            args.npy, args.acq = True, False
            if use_name:
                args.infile = os.path.join(args.srcdir, args.name + "_x.npy")
                args.targetfile = os.path.join(args.srcdir, args.name + "_y.npy")
                args.metafile = os.path.join(args.srcdir, args.name + "_meta.tsv")
            if not os.path.exists(args.infile):
                raise ValueError('Input data file "{}" does not exist'.format(args.infile))
            if not os.path.exists(args.targetfile):
                raise ValueError('Input targets file "{}" does not exist'.format(args.targetfile))
            if args.metafile != None and not os.path.exists(args.metafile):
                raise ValueError('Input metadata file "{}" does not exist'.format(args.metafile))
        else:
            ## single acqfile with optional triggerfile
            args.npy, args.acq = False, True
            if not os.path.exists(args.acqfile):
                raise ValueError('Input acquisition file "{}" does not exist'.format(args.acqfile))
            if args.triggerfile != None and not os.path.exists(args.triggerfile):
                raise ValueError('Input trigger file "{}" does not exist'.format(args.triggerfile))
        # Check network modules and trained models built from those modules
        for network in args.networks:
            network_name = network[0]
            model_file = network[1]
            if not os.path.exists(model_file + ".meta"):
                raise ValueError('Model file {} for network {} does not exist: '.format(model_file, network_name))

        # Check logdir if not using default
        if args.logdir != self.default_logdir and not os.path.exists(args.logdir):
            raise ValueError('Non-default logging output directory does not exist')

        return args
