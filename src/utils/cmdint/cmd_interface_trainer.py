import os
import argparse
from .base_cmd_interface import base_cmd_interface as bci

class cmd_interface(bci):

    def __init__(self):
        bci.__init__(self)
        self.default_logdir = os.path.join('/run/user/', str(os.getuid()), 'convnet_euso/')
        self.parser = argparse.ArgumentParser(description="Train candidate network(s) with simulated data")

        # data and logging configuration parameters
        self.parser.add_argument('-i', '--infile',
                                help=('the npy file storing simulated air shower data. If specified, implicitly overrides'
                                      ' --params, otherwise --params is used to construct the input file name. Must be'
                                      ' specified alongside --targetfile'))
        self.parser.add_argument('-t', '--targetfile',
                                help=('the npy file storing expected neural network outputs (targets) for the data.' 
                                      ' If specified, implicitly overrides --params, otherwise --params is used'
                                      ' to construct the target file name. Must be specified alongside --infile'))
        self.parser.add_argument('-p', '--params', metavar=self._param_metavars, type=int, nargs=len(self._param_metavars),
                                help=('parameters of input data to use; if not explicitly stated via -i and -t options,'
                                      ' the input data and targets file names are constructed from these parameters'))
        self.parser.add_argument('-s', '--srcdir',
                                help=('directory containing input data and target files; only required if --params is used,'
                                      ' otherwise it is completely ignored'))
        self.parser.add_argument('-l', '--logdir', default=self.default_logdir,
                                help=('directory to store output logs, default: "/run/user/$USERID/convnet_euso/".'
                                      ' If a non-default directory is used, it must exist prior to calling this script,' 
                                      ' otherwise an error will be thrown'))

        # training configuration parameters (nunmber of epochs, learning rate etc.)
        self.parser.add_argument('-n', '--networks', action='append', required=True, metavar='NETWORK_NAME',
                                help='names of network modules to use')
        self.parser.add_argument('-e', '--epochs', type=self._positive_int, default=11,
                                help='number of training epochs per network')
        self.parser.add_argument('--learning_rate', type=self._positive_float,
			        help='learning rate for all the tested networks')
        self.parser.add_argument('--optimizer',
                                help='gradient descent optimizer for all the tested networks')
        self.parser.add_argument('--loss', 
                                help='loss function to use for all the tested networks')

        # Misc
        self.parser.add_argument('--save', action='store_true', 
                                help=('save the model after training. The model files are saved as'
                                      ' logdir/current_time/network_name/network_name.tflearn*'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        pa_defined = (args.params != None)
        sd_defined = (args.srcdir != None)
        if_defined = (args.infile != None)
        tf_defined = (args.targetfile != None)
        # Error checking
        ## only specified one of the two input file names, error
        if if_defined != tf_defined:
            raise ValueError('both input and target file names must be specified unless using params/srcdir combination')
    
        # or if not tf_defined; does not make a difference here since either both are defined or neither
        if not if_defined:
            ## either params or srcdir are not specified, error
            if not pa_defined or not sd_defined:
                raise ValueError('params and srcdir must be defined if input and target files are not explicitly provided')            
            data, targets = self.params_to_filenames(args.srcdir, args.params)
            args.infile = data + '.npy'
            args.targetfile = targets + '.npy'

        if not os.path.exists(args.infile):
            raise ValueError('Input data file does not exist')
        if not os.path.exists(args.targetfile):
            raise ValueError('Input target file does not exist')
        if args.logdir != self.default_logdir and not os.path.exists(args.logdir):
            raise ValueError('Non-default logging output directory does not exist')

        return args

