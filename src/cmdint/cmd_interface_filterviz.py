import os
import sys
import argparse

import cmdint.common.args as cargs
import cmdint.common.network_args as net_args

class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Visualize convolutional filters of trained model")
        parser.add_argument('logdir',
                            help=('Directory to output visualized filter '
                                  'images to.'))

        # trained neural network model
        group = parser.add_argument_group('Neural network settings')
        net_args.add_network_arg(group, short_alias='n')
        net_args.add_model_file_arg(group, short_alias='m', required=True)
        packet_args = cargs.packet_args(long_alias='packet_dims')
        packet_args.helpstr = 'Dimensions of packets used for input items'
        packet_args.add_packet_arg(group, short_alias='p')

        # visualization settings
        group = parser.add_argument_group('Visualization settings')
        group.add_argument('--start_filter', default=0, type=int,
                           help=('index of first filter to visualize.'))
        group.add_argument('--stop_filter', default=None, type=int,
                           help=('index of the filter after the last filter '
                                 'to visualize.'))
        group.add_argument('--start_depth', default=0, type=int,
                           help=('first depth index of filters to visualize.'))
        group.add_argument('--stop_depth', default=None, type=int,
                           help=('index after the last depth index of filters '
                                 'to visualize.'))

        # misc
        parser.add_argument('--usecpu', action='store_true',
                            help=('Use host CPU instead of the CUDA device. '
                                  'On systems without a dedicated CUDA device '
                                  'and no CUDA-enabled version  of tensorflow '
                                  'installed, this flag has no effect.'))

        self.packet_args = packet_args
        self.parser = parser

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        logdir = args.logdir
        if not os.path.isdir(logdir):
            raise Exception('Invalid logdir: {}'.format(logdir))
        template = self.packet_args.packet_arg_to_template(args)

        args_dict = {}
        args_dict['network'] = args.network
        args_dict['model_file'] = args.model_file
        args_dict['usecpu'] = args.usecpu
        args_dict['logdir'] = logdir
        args_dict['packet_shape'] = template.packet_shape
        args_dict['filter_slice'] = slice(args.start_filter, args.stop_filter)
        args_dict['depth_slice'] = slice(args.start_depth, args.stop_depth)

        return args_dict
