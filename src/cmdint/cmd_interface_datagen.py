import os
import argparse

import utils.synth_data_utils as sdutils

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Create simulated air shower data as numpy arrays")
        self.params_args = sdutils.params_args()
        self.shower_args = sdutils.shower_args()
        self.params_args.add_packet_cmd_args(self.parser)
        self.params_args.add_other_cmd_args(self.parser)
        self.shower_args.add_cmd_args(self.parser)

        self.parser.add_argument('--num_shuffles', type=int, default=1,
                                help=('Number of times the generated data should be shuffled randomly after creation'))
        self.parser.add_argument('-d', '--destdir', default=os.path.curdir,
                                help=('The directory in which to store output and target files, defaults to current'
                                     ' working directory.'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        self.params_args.check_cmd_args(args)
        args.template = self.params_args.args_to_packet_template(args)

        if args.num_shuffles < 0:
            raise ValueError('Number of times the data is shuffled cannot be negative')
        if not os.path.exists(args.destdir):
            os.mkdir(args.destdir)
            raise ValueError('The output directory {} does not exist'.format(args.destdir))

        # set default value ranges for start coordinates of shower values, if user has not set them
        # let start coordinates be at least a distance of 3/4 * duration from the edges of the frame
        limit = int(3*args.duration[1]/4)
        args.start_gtu = (0, args.template.num_frames - limit) if args.start_gtu == None else args.start_gtu
        args.start_x = (limit, args.template.frame_width - limit) if args.start_x == None else args.start_x
        args.start_y = (limit, args.template.frame_height - limit) if args.start_y == None else args.start_y
        self.shower_args.check_cmd_args(args, args.template)
        args.shower_properties = self.shower_args.args_to_dict(args)

        # set default filenames for data and targets
        params_impl = self.params_args.args_to_string(args)
        shower_impl = self.shower_args.args_to_string(args)
        # TODO: Might want to use a better way to create dataset files (maybe keep metadata 
        # such as shower and packet properties in an sqlite database and only distinguish files
        # by a string representing its creation timestamp)
        xy_impl = os.path.join(args.destdir, ('simu_data__{}__{}_yx.npy'.format(params_impl, shower_impl)))
        xgtu_impl = os.path.join(args.destdir, ('simu_data__{}__{}_gtux.npy'.format(params_impl, shower_impl)))
        ygtu_impl = os.path.join(args.destdir, ('simu_data__{}__{}_gtuy.npy'.format(params_impl, shower_impl)))
        targetfile_impl = os.path.join(args.destdir, ('simu_data__{}__{}_targets.npy'.format(params_impl, shower_impl)))

        args.outfiles = (xy_impl, xgtu_impl, ygtu_impl)
        args.targetfile =  targetfile_impl

        return args
