import argparse
from .base_cmd_interface import base_cmd_interface

class cmd_interface(base_cmd_interface):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Test candidate networks with simulated data")
        self.parser.add_argument('width', type=self.min_dim_size, 
                                help='width of all frames of simulated air shower data')
        self.parser.add_argument('height', type=self.min_dim_size, 
                                help='height of all frames of simulated air shower data')
        self.parser.add_argument('num_frames', type=self.positive_int, 
                                help='number of frames of simulated air shower data')
        self.parser.add_argument('lam', type=self.unsigned_int, 
                                help='mean value of backgroud noise (lambda in Poisson distributions)')
        self.parser.add_argument('bg_diff', type=self.positive_int, 
                                help='integer difference between mean value of background noise (lambda) and shower pixels')
        self.parser.add_argument('-e', '--epochs', type=self.positive_int, default=11,
                                help='number of training epochs per network')
        self.parser.add_argument('-n', '--networks', action='append', required=True, 
                                help='names of network modules to use')
        self.parser.add_argument('-s', '--srcdir', 
                                help='name of the source directory from which to retrieve data')
        self.parser.add_argument('-i', '--infile', 
                                help='name of the npy file storing simulated air shower data, default value is constructed from passed in positional arguments and source directory')
        self.parser.add_argument('-t', '--targetfile', 
                                help='name of the npy file storing expected output values for given air shower data, default value is constructed from passed in positional arguments and source directory')
        self.parser.add_argument('--save', action='store_true')

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)
        
        if_defined = (args.infile != None)
        tf_defined = (args.targetfile != None)
        # only specified one of the two input file names, error
        if if_defined != tf_defined:
            raise ValueError("both input and target file names must be specified when not using default values")

        # or if !tf_defined, does not make a difference here
        if not if_defined:
            args.infile = 'simu_data_x_{}_{}_{}_bg_{}_diff_{}'.format(args.num_frames, args.width, args.height, args.lam, args.bg_diff)
            args.targetfile =  'simu_data_y_{}_{}_{}_bg_{}_diff_{}'.format(args.num_frames, args.width, args.height, args.lam, args.bg_diff)
        #if args.destdir != None:
        #    args.infile = args.srcdir.rstrip("/") + "/" + args.infile
        #    args.targetfile = args.srcdir.rstrip("/") + "/" + args.targetfile

        return args
