import os
import argparse

import cmdint.common_args as common_args
import utils.synth_data_utils as sdutils

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Create simulated air shower data as numpy arrays")

        # packet dimensions
        common_args.add_packet_args(self.parser, required=True)
        # arguments qualifying shower property ranges
        self.parser.add_argument('--shower_max', metavar=('MIN', 'MAX'), nargs=2, type=int, required=True,
                                help=('Relative difference between pixel values of shower track and background. This is'
                                    ' a range of values from MIN to MAX, inclusive. If MIN == MAX, for all packets'
                                    ' the shower line has the same peak potential intensity.'))
        self.parser.add_argument('--duration', metavar=('MIN', 'MAX'), nargs=2, type=int, required=True,
                                help=('Duration of shower tracks in number of GTU or frames containing shower pixels.'
                                    ' The actual duration of a shower for any data item is from MIN to MAX, inclusive.'
                                    ' If MIN == MAX, the duration is always the same.'))
        self.parser.add_argument('--start_gtu', metavar=('MIN', 'MAX'), nargs=2, type=int,
                                help=('First GTU containing shower pixels. This is a range of GTUs from MIN to MAX, inclusive,'
                                    ' where a simulated shower line begins. If MIN == MAX, for all packets the shower line'
                                    ' starts at the same GTU in a packet.'))
        self.parser.add_argument('--start_y', metavar=('MIN', 'MAX'), nargs=2, type=int,
                                help=('The y coordinate of a packet frame at which a shower line begins. This is a range'
                                    ' of y coordinate values from MIN to MAX, inclusive, where a shower line can begin.'
                                    ' If MIN == MAX, all packets have shower lines starting at the same y coordinate.'))
        self.parser.add_argument('--start_x', metavar=('MIN', 'MAX'), nargs=2, type=int,
                                help=('The x coordinate of a packet frame at which a shower line begins. This is a range'
                                    ' of x coordinate values from MIN to MAX, inclusive, where a shower line can begin.'
                                    ' If MIN == MAX, all packets have shower lines starting at the same x coordinate.'))

        # additional arguments applying to generated dataset
        self.parser.add_argument('--num_data', required=True, type=int,
                            help=('Number of data items (both noise and shower), corresponds to number of packets'))
        self.parser.add_argument('--bg_lambda', required=True, type=float,
                            help=('Average of background pixel values (lambda in Poisson distributions'))
        self.parser.add_argument('--bad_ECs', type=int, nargs=2, metavar=('MIN', 'MAX'), default=(0, 0),
                            help=('Number of malfunctioned EC modules in the data. The actual number of such ECs'
                                ' in any data item is from MIN to MAX, inclusive. If MIN == MAX, the number of bad ECs'
                                ' is an exact number, barring cases where keeping this requirement would knock out ECs'
                                ' containing shower pixels. Default value range: (0, 0).'))
        self.parser.add_argument('--num_shuffles', type=int, default=1,
                                help=('Number of times the generated data should be shuffled randomly after creation'))

        # information about which output files to create
        common_args.add_output_type_dataset_args(self.parser)

        # output directory
        self.parser.add_argument('-d', '--destdir', default=os.path.curdir,
                                help=('The directory in which to store output and target files, defaults to current'
                                     ' working directory.'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        packet_templ = common_args.packet_args_to_packet_template(args)
        args.packet_template = packet_templ
        packet_str = common_args.packet_args_to_string(args)

        sx, sy, sgtu = args.start_x, args.start_y, args.start_gtu
        smax, dur = args.shower_max, args.duration
        shower_templ = sdutils.simulated_shower_template(packet_templ, dur, smax, start_gtu=sgtu,
                                                        start_y=sy, start_x=sx)
        args.shower_template = shower_templ
        sx, sy, sgtu = shower_templ.start_x, shower_templ.start_y, shower_templ.start_gtu
        shower_str = 'shower_gtu_{}-{}_y_{}-{}_x_{}-{}_duration_{}-{}_bgdiff_{}-{}'.format(
                        sgtu[0], sgtu[1], sy[0], sy[1], sx[0], sx[1], dur[0], dur[1], smax[0], smax[1])

        n_data = args.num_data
        lam = args.bg_lambda
        bec_min, bec_max = args.bad_ECs[0:2]
        dataset_str = 'num_{}_bad_ecs_{}-{}_lam_{}'.format(n_data, bec_min, bec_max, lam)

        if args.num_shuffles < 0:
            raise ValueError('Number of times the data is shuffled cannot be negative')
        if not os.path.exists(args.destdir):
            os.mkdir(args.destdir)
            raise ValueError('The output directory {} does not exist'.format(args.destdir))

        args.dataset_helper = common_args.output_type_dataset_args_to_helper(args)

        # set default filenames for data and targets
        # TODO: Might want to use a better way to create dataset files (maybe keep metadata
        # such as shower and packet properties in an sqlite database and only distinguish files
        # by a string representing its creation timestamp)
        common_filename_part = os.path.join(args.destdir, ('simu_data__{}__{}__{}'.format(packet_str, shower_str, dataset_str)))
        outfiles, targetfile = common_args.output_type_dataset_args_to_filenames(args, common_filename_part)

        args.outfiles = outfiles
        args.targetfile =  targetfile

        return args

    # def args_to_dict(self, args):
    #     return {'start_gtu': args.start_gtu, 'start_x': args.start_x, 'start_y': args.start_y,
    #             'bg_diff': args.bg_diff, 'duration': args.duration}