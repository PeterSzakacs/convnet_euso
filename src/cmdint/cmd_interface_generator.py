import os
import argparse

import cmdint.common.argparse_types as atypes
import cmdint.common.args as cargs
import cmdint.common.dataset_args as dargs
import utils.data_templates as templates

class CmdInterface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Create simulated air shower data as numpy arrays")
        out_aliases = {'dataset name': 'name', 'dataset directory': 'outdir'}
        self.packet_args = cargs.PacketArgs()
        self.dset_args = dargs.DatasetArgs(output_aliases=out_aliases)
        self.item_args = dargs.ItemTypeArgs()

        ds_group = self.parser.add_argument_group('dataset configuration')
        # packet dimensions
        self.packet_args.add_packet_arg(ds_group, required=True)
        # output dataset
        atype = dargs.arg_type.OUTPUT
        self.dset_args.add_dataset_arg_double(ds_group, atype)
        self.item_args.add_item_type_args(ds_group, atype)
        ds_group.add_argument('--num_data', required=True,
                              type=atypes.int_range(1),
                              help=('Number of data items (both noise and '
                                    'shower)'))
        ds_group.add_argument('--dtype', default='uint8',
                              help=('Data type of dataset items (default: '
                                    'uint8)'))

        shower_group = self.parser.add_argument_group('shower properties')
        # arguments qualifying shower property ranges
        args  = ['shower_max', 'duration', 'track_length', 'start_gtu',
                 'start_y', 'start_x']
        reqs  = [True, True, True, False, False, False]
        descs = [
            'Peak relative diff. between shower track and bg pixel values',
            'Number of GTU or frames containing shower track pixels',
            'Length of shower tracks as viewed in the yx projection',
            'First GTU or packet frame containing shower pixels',
            'Start_gtu frame X coordinate from which the shower tracks begin',
            'Start_gtu frame Y coordinate from which the shower tracks begin'
        ]
        types = ([atypes.int_range(1)] * 3) + ([atypes.int_range(0)] * 3)
        for idx in range(len(args)):
            arg = args[idx]
            cargs.add_number_range_arg(shower_group, arg, arg_desc=descs[idx],
                                       required=reqs[idx], arg_type=types[idx])

        bg_group = self.parser.add_argument_group('background properties')
        # additional arguments applying to packet background
        cargs.add_number_range_arg(bg_group, 'bg_lambda', required=True,
                                   arg_type=atypes.float_range(0),
                                   arg_desc=('Bg pixel values average '
                                             '(Poisson distribution lambda)'))
        cargs.add_number_range_arg(bg_group, 'bad_ECs', default=(0, 0),
                                   arg_type=atypes.int_range(-1),
                                   arg_desc='Bad ECs count per data item')

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

        atype = dargs.arg_type.OUTPUT
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