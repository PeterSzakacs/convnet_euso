import os
import argparse

import cmdint.common.argparse_types as atypes
import cmdint.common.dataset_args as dargs
import net.constants as net_cons

class cmd_interface():

    def __init__(self):
        self.default_logdir = net_cons.DEFAULT_TRAIN_LOGDIR
        parser = argparse.ArgumentParser(
            description="Train convolutional network(s) using provided dataset")

        # dataset input
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        dset_args = dargs.dataset_args(input_aliases=in_aliases)
        item_args = dargs.item_types_args()
        atype = dargs.arg_type.INPUT
        group = parser.add_argument_group(title="Input dataset")
        dset_args.add_dataset_arg_double(group, atype)
        item_args.add_item_type_args(group, atype)
        group.add_argument('--test_items_count', type=atypes.int_range(1),
                           help='Number of dataset items to include in the '
                                'test set. Overrides test_items_fraction.')
        group.add_argument('--test_items_fraction', type=float, default=0.1,
                           help='Number of dataset items to include in the '
                                'test set, expressed as a fraction.')
        modes = net_cons.DATASET_SPLIT_MODES
        group.add_argument('--split_mode', choices=modes, required=True,
                           help='Method of splitting the test items subset '
                                'from the input dataset.')

        # network(s) to train
        group = parser.add_argument_group(title="Network configuration")
        group.add_argument('-n', '--networks', action='append', required=True,
                           metavar='NETWORK_NAME',
                           help='names of network modules to use')
        group.add_argument('--logdir', default=self.default_logdir,
                           help=('directory to store training logs. If a '
                                 'non-default directory is used, it must '
                                 'exist prior to calling this script'))
        group.add_argument('--save', action='store_true',
                           help=('save the model after training. Model files '
                                 'are saved under the logdir as net.network_'
                                 'name/net.network_name.tflearn.*'))

        # training settings
        group = parser.add_argument_group(title="Training parameters")
        group.add_argument('-e', '--epochs', default=11,
                                type=atypes.int_range(1),
                                help='number of training epochs per network')
        group.add_argument('--learning_rate', type=float,
                           help='learning rate for all the tested networks')
        group.add_argument('--optimizer',
                           help=('gradient descent optimizer to use for all '
                                 'the tested networks'))
        group.add_argument('--loss',
                           help=('loss function to use for all the tested '
                                 'networks'))
        self.parser = parser
        self.dset_args = dset_args
        self.item_args = item_args

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        atype = dargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)
        args.item_types = self.item_args.get_item_types(args, atype)
        logdir = args.logdir
        if logdir != self.default_logdir and not os.path.isdir(logdir):
            raise ValueError(('Invalid non-default logging directory: {}'
                              .format(logdir)))

        return args
