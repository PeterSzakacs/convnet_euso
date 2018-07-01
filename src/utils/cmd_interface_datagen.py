import argparse
from .base_cmd_interface import base_cmd_interface

class cmd_interface(base_cmd_interface):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Create simulated air shower data as numpy arrays")
        self.parser.add_argument('width', type=self.min_dim_size, 
                                help='width of all frames of simulated air shower data')
        self.parser.add_argument('height', type=self.min_dim_size, 
                                help='height of all frames of simulated air shower data')
        self.parser.add_argument('num_frames', type=self.positive_int, 
                                help='number of frames of simulated ar shower data')
        self.parser.add_argument('lam', type=self.unsigned_int, 
                                help='mean value of backgroud noise (lambda in Poisson distributions)')
        self.parser.add_argument('bg_diff', type=self.positive_int, 
                                help='integer difference between mean value of background noise (lambda) and shower pixels')
        self.parser.add_argument('-o', '--outfile', 
                                help='name of the npy file to store simulated air shower data')
        self.parser.add_argument('-t', '--targetfile', 
                                help='name of the npy file to store expected processing output values for given air shower data')
        self.parser.add_argument('-d', '--destdir', 
                                help='name of the target directory in which to store output and target files')

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        if args.outfile == None:
            args.outfile = 'simu_data_x_{}_{}_{}_bg_{}_diff_{}'.format(args.num_frames, args.width, args.height, args.lam, args.bg_diff)
        if args.targetfile == None:
            args.targetfile =  'simu_data_y_{}_{}_{}_bg_{}_diff_{}'.format(args.num_frames, args.width, args.height, args.lam, args.bg_diff)
        if args.destdir != None:
            args.outfile = args.destdir.rstrip("/") + "/" + args.outfile
            args.targetfile = args.destdir.rstrip("/") + "/" + args.targetfile

        return args
