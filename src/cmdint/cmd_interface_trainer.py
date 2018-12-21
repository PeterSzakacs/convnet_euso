import os
import argparse

import cmdint.common_args as cargs

class cmd_interface():

    def __init__(self):
        self.default_logdir = os.path.join('/run/user/', str(os.getuid()), 'convnet_trainer/')
        self.parser = argparse.ArgumentParser(
            description="Train network(s) using the provided dataset")
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        self.dset_args = cargs.dataset_args(input_aliases=in_aliases)
        self.item_args = cargs.item_types_args()

        # dataset input
        atype = cargs.arg_type.INPUT
        self.dset_args.add_dataset_arg_double(self.parser, atype)
        self.item_args.add_item_type_args(self.parser, atype)

        # network(s) to train
        self.parser.add_argument('-n', '--networks', action='append', required=True, metavar='NETWORK_NAME',
                                help='names of network modules to use')

        # training configuration parameters (nunmber of epochs, learning rate etc.)
        self.parser.add_argument('-e', '--epochs', type=int, default=11,
                                help='number of training epochs per network')
        self.parser.add_argument('--learning_rate', type=float,
			                    help='learning rate for all the tested networks')
        self.parser.add_argument('--optimizer',
                                help='gradient descent optimizer for all the tested networks')
        self.parser.add_argument('--loss',
                                help='loss function to use for all the tested networks')

        # Misc
        self.parser.add_argument('--logdir', default=self.default_logdir,
                                help=('directory to store output logs, default: "/run/user/$USERID/convnet_trainer/".'
                                      ' If a non-default directory is used, it must exist prior to calling this script,'
                                      ' otherwise an error will be thrown'))
        self.parser.add_argument('--save', action='store_true',
                                help=('save the model after training. The model files are saved as'
                                      ' logdir/current_time/network_name/network_name.tflearn*'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = cargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)
        if args.logdir != self.default_logdir and not os.path.isdir(args.logdir):
            raise ValueError(('Invalid non-default logging directory: {}'
                              .format(args.logdir)))

        return args

