import os
import argparse

import cmdint.argparse_types as atypes
import cmdint.common_args as cargs
import utils.data_templates as templates

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Create simulated air shower data as numpy arrays")
        out_aliases = {'dataset name': 'name', 'dataset directory': 'outdir'}
        self.packet_args = cargs.packet_args()
        self.dset_args = cargs.dataset_args(output_aliases=out_aliases)
        self.item_args = cargs.item_types_args()

        # packet dimensions
        self.packet_args.add_packet_arg(self.parser, required=True)

        # output dataset
        atype = cargs.arg_type.OUTPUT
        self.dset_args.add_dataset_arg_double(self.parser, atype)
        self.item_args.add_item_type_args(self.parser, atype)

        self.parser.add_argument('--num_data', required=True, type=atypes.int_range(1),
                            help=('Number of data items (both noise and shower), corresponds to number of packets'))


        # arguments qualifying shower property ranges
        self.parser.add_argument('--shower_max', metavar=('MIN', 'MAX'), nargs=2, type=atypes.int_range(1), required=True,
                                help=('Relative difference between pixel values of shower track and background. This is'
                                    ' a range of values from MIN to MAX, inclusive. If MIN == MAX, for all packets'
                                    ' the shower line has the same peak potential intensity.'))
        self.parser.add_argument('--duration', metavar=('MIN', 'MAX'), nargs=2, type=atypes.int_range(1), required=True,
                                help=('Duration of shower tracks in number of GTU or frames containing shower pixels.'
                                    ' The actual duration of a shower for any data item is from MIN to MAX, inclusive.'
                                    ' If MIN == MAX, the duration is always the same.'))
        self.parser.add_argument('--track_length', metavar=('MIN', 'MAX'), nargs=2, type=atypes.int_range(1), required=True,
                                help=('Length of shower tracks as viewed in the yx projection of the packets.'
                                    ' The actual length of a track for any data item is from MIN to MAX, inclusive.'
                                    ' If MIN == MAX, the length is always the same.'))
        self.parser.add_argument('--start_gtu', metavar=('MIN', 'MAX'), nargs=2, type=atypes.int_range(0),
                                help=('First GTU containing shower pixels. This is a range of GTUs from MIN to MAX, inclusive,'
                                    ' where a simulated shower line begins. If MIN == MAX, for all packets the shower line'
                                    ' starts at the same GTU in a packet.'))
        self.parser.add_argument('--start_y', metavar=('MIN', 'MAX'), nargs=2, type=atypes.int_range(0),
                                help=('The y coordinate of a packet frame at which a shower line begins. This is a range'
                                    ' of y coordinate values from MIN to MAX, inclusive, where a shower line can begin.'
                                    ' If MIN == MAX, all packets have shower lines starting at the same y coordinate.'))
        self.parser.add_argument('--start_x', metavar=('MIN', 'MAX'), nargs=2, type=atypes.int_range(0),
                                help=('The x coordinate of a packet frame at which a shower line begins. This is a range'
                                    ' of x coordinate values from MIN to MAX, inclusive, where a shower line can begin.'
                                    ' If MIN == MAX, all packets have shower lines starting at the same x coordinate.'))

        # additional arguments applying to packet background
        self.parser.add_argument('--bg_lambda', metavar=('MIN', 'MAX'), nargs=2, type=atypes.float_range(0), required=True,
                            help=('Average of background pixel values (lambda in Poisson distributions). The actual'
                                ' background mean in any data item is from MIN to MAX, inclusive. If MIN == MAX, the'
                                ' background mean is constant and the same in all packets from which itmes are created.'))
        self.parser.add_argument('--bad_ECs', metavar=('MIN', 'MAX'), nargs=2, type=atypes.int_range(-1), default=(0, 0),
                            help=('Number of malfunctioned EC modules in the data. The actual number of such ECs'
                                ' in any data item is from MIN to MAX, inclusive. If MIN == MAX, the number of bad ECs'
                                ' is an exact number, barring cases where keeping this requirement would knock out ECs'
                                ' containing shower pixels. Default value range: (0, 0).'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        packet_templ = self.packet_args.packet_arg_to_template(args)
        packet_str = self.packet_args.packet_arg_to_string(args)
        args.packet_template = packet_templ

        sx, sy, sgtu = args.start_x, args.start_y, args.start_gtu
        smax, dur, length = args.shower_max, args.duration, args.track_length
        shower_templ = templates.simulated_shower_template(
            packet_templ, dur, smax, length, start_gtu=sgtu, start_y=sy, start_x=sx
        )
        sx, sy, sgtu = shower_templ.start_x, shower_templ.start_y, shower_templ.start_gtu
        shower_str = 'shower_gtu_{}-{}_y_{}-{}_x_{}-{}_duration_{}-{}_len_{}-{}_bgdiff_{}-{}'.format(
                     sgtu[0], sgtu[1], sy[0], sy[1], sx[0], sx[1], dur[0], dur[1], length[0], length[1], smax[0], smax[1])
        args.shower_template = shower_templ

        n_data, lam, bec = args.num_data, args.bg_lambda, args.bad_ECs
        dataset_str = 'num_{}_bad_ecs_{}-{}_lam_{}-{}'.format(
            n_data, bec[0], bec[1], lam[0], lam[1]
        )
        args.bg_template = templates.synthetic_background_template(
            packet_templ, bg_lambda=lam, bad_ECs_range=bec
        )

        if not os.path.isdir(args.outdir):
            raise ValueError("Invalid output directory {}".format(args.outdir))

        atype = cargs.arg_type.OUTPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)

        # TODO: Might want to use a better way to create dataset files (maybe keep metadata
        # such as shower and packet properties in an sqlite database and only distinguish files
        # by a string representing its creation timestamp)
        args.name = args.name or 'simu_data__{}__{}__{}'.format(
            packet_str, shower_str, dataset_str
        )

        return args

    # def args_to_dict(self, args):
    #     return {'start_gtu': args.start_gtu, 'start_x': args.start_x, 'start_y': args.start_y,
    #             'bg_diff': args.bg_diff, 'duration': args.duration}