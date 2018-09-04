import os
import argparse
from .base_cmd_interface import base_cmd_interface as bci

class cmd_interface(bci):

    def __init__(self):
        bci.__init__(self)
        self.parser = argparse.ArgumentParser(description="Create simulated air shower data as numpy arrays")
        self.parser.add_argument('-p', '--params', metavar=self._param_metavars, type=int, nargs=len(self._param_metavars), 
                                required=True,
                                help=('parameters of generated data to use. The default file names'
                                     ' for output data and targets are constructed from these,'
                                     ' unless explicitly stated with --outfile and --targetfile'))
        self.parser.add_argument('--duration', type=self._positive_int,
                                help=('duration of shower tracks in number of gtu or frames in which shower pixels'
                                     ' are located. If not specified, uses shower tracks of lengths from 3 to 16 GTU.'))
        self.parser.add_argument('-o', '--outfile', 
                                help=('name of the npy file (sans .npy extension) to store simulated air shower data.'
                                     ' If specified, overrides using values of --params and --destdir to construct'
                                     ' the file name'))
        self.parser.add_argument('-t', '--targetfile', 
                                help=('name of the npy file (sans .npy extension) to store expected output (targets)'
                                     ' for simulated air shower data. If specified, overrides using values of --params'
                                     ' and --destdir to construct the file name'))
        self.parser.add_argument('-d', '--destdir', default=os.path.curdir,
                                help=('the directory in which to store output and target files, defaults to current'
                                     ' working directory. Only ever used if --outfile and --targetfile are unspecified,' 
                                     ' otherwise ignored'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        of_defined = (args.outfile != None)
        tf_defined = (args.targetfile != None)
        if of_defined != tf_defined:
            raise ValueError("Error, either specify both data and targets file or let the script use their default values")

        # set default filenames for data and targets if not specified
        datafile_impl, targetfile_impl = self.params_to_filenames(args.destdir, args.params)
        # or if not tf_defined, does not matter
        if not of_defined:
            args.outfile = datafile_impl
            args.targetfile =  targetfile_impl
        if args.duration and args.duration < 3:
            raise ValueError("Error, shower duration must be at least 3 GTU")

        # not strictly necessary, but easier to use these values later in the data generation script 
        # (otherwise would have to access using args.params[index] instead of args.param_name)
        params = args.params
        args.width, args.height, args.num_frames = params[0], params[1], params[2]
        args.lam, args.bg_diff = params[3], params[4]
        args.num_merged = params[5]

        return args
