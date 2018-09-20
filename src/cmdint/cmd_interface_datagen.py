import os
import argparse
from .base_cmd_interface import base_cmd_interface as bci

class cmd_interface(bci):

    def __init__(self):
        bci.__init__(self)
        self._bad_ECs_default = (0, 0)
        self._duration_default = (3, 15)

        self.parser = argparse.ArgumentParser(description="Create simulated air shower data as numpy arrays")
        self.parser.add_argument('-p', '--params', metavar=self._param_metavars, type=int, nargs=len(self._param_metavars), 
                                required=True,
                                help=('parameters of generated data to use. The default file names'
                                     ' for output data and targets are constructed from these,'
                                     ' unless explicitly stated with --outfile and --targetfile'))
        self.parser.add_argument('--bad_ECs', nargs=2, metavar=('MIN', 'MAX'), default=(0, 0), type=int,
                                help=('simulate the effect of malfunctioned EC modules in the data. The actual number of such ECs'
                                    ' in any data item is from MIN to MAX, inclusive. If MIN == MAX, simulates an exact number of EC'
                                    ' modules as bad. Default value range: {}'.format(self._bad_ECs_default)))
        self.parser.add_argument('--duration', nargs=2, metavar=('MIN', 'MAX'), default=(0, 0), type=self._positive_int,
                                help=('duration of shower tracks in number of GTU or frames contatining shower pixels.'
                                    ' The actual duration of a shower for any data item is from MIN to MAX, inclusive.'
                                    ' If MIN == MAX, the duration is always the same. Default value range: {}'.format(self._duration_default)))
        self.parser.add_argument('--num_shuffles', type=self._positive_int, default=1,
                                help=('number of times the generated data should be shuffled randomly after creation'))
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
        if args.duration != self._duration_default:
            min, max = args.duration[0], args.duration[1]
            if (min > max or min < 0 or max < 0):
                raise ValueError("Error, invalid range for shower duration {}".format(args.duration))
        if args.bad_ECs != self._bad_ECs_default:
            min, max = args.bad_ECs[0], args.bad_ECs[1]
            if (min > max or min < 0 or max < 0):
                raise ValueError("Error, invalid range for number of malfunctioned EC modules {}".format(args.bad_ECs))

        # not strictly necessary, but easier to use these values later in the data generation script 
        # (otherwise would have to access using args.params[index] instead of args.param_name)
        params = args.params
        args.width, args.height, args.num_frames = params[0], params[1], params[2]
        args.lam, args.bg_diff = params[3], params[4]
        args.num_merged = params[5]

        return args
