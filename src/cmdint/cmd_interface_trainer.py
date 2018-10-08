import os
import argparse

import cmdint.common_args as common_args

class cmd_interface():

    def __init__(self):
        self.default_logdir = os.path.join('/run/user/', str(os.getuid()), 'convnet_trainer/')
        self.parser = argparse.ArgumentParser(description="Train candidate network(s) with simulated data")
        common_args.add_input_type_dataset_args(self.parser)
        common_args.add_packet_args(self.parser)

        # data and logging configuration parameters
        self.parser.add_argument('--name', required=True,
                                help=('common part of name of input data files; the input data and targets file names'
                                      ' are constructed from these parameters'))
        self.parser.add_argument('--srcdir', required=True,
                                help=('directory containing input data and target files'))
        self.parser.add_argument('-l', '--logdir', default=self.default_logdir,
                                help=('directory to store output logs, default: "/run/user/$USERID/convnet_trainer/".'
                                      ' If a non-default directory is used, it must exist prior to calling this script,'
                                      ' otherwise an error will be thrown'))

        # training configuration parameters (nunmber of epochs, learning rate etc.)
        self.parser.add_argument('-n', '--networks', action='append', required=True, metavar='NETWORK_NAME',
                                help='names of network modules to use')
        self.parser.add_argument('-e', '--epochs', type=int, default=11,
                                help='number of training epochs per network')
        self.parser.add_argument('--learning_rate', type=float,
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

        common_filename_part = os.path.join(args.srcdir, args.name)
        infiles, targetfile = common_args.input_type_dataset_args_to_filenames(
            args, common_filename_part
        )
        args.infiles, args.targetfile = infiles, targetfile

        packet_template = common_args.packet_args_to_packet_template(args)
        print(packet_template.packet_shape)
        dataset_helper = common_args.input_type_dataset_args_to_helper(args)
        input_shapes = dataset_helper.get_data_item_shapes(
            packet_template.packet_shape
        )
        args.input_shapes = {k:((None, *v, 1) if v != None else None)
                             for k,v in input_shapes.items()}

        for filename in args.infiles:
            if not os.path.exists(filename):
                raise ValueError('Input data file {} does not exist'.format(filename))
        if not os.path.exists(args.targetfile):
            raise ValueError('Input target file does not exist')
        if args.logdir != self.default_logdir and not os.path.exists(args.logdir):
            raise ValueError('Non-default logging output directory does not exist')

        return args

