import os
import argparse

import net.constants as net_cons
import cmdint.common.argparse_types as atypes
import cmdint.common.dataset_args as dargs

class cmd_interface():

    def __init__(self):
        self.default_logdir = net_cons.DEFAULT_XVAL_LOGDIR
        parser = argparse.ArgumentParser(
            description="Perform Kfold cross-validation on the passed in "
                        "convolutional neural network with the given dataset.")

        # cross-validation settings
        group = parser.add_argument_group(title="Cross-validation parameters")
        group.add_argument('--num_crossvals', type=atypes.int_range(1),
                           required=True,
                           help='number of cross validations to perform')
        group.add_argument('-e', '--epochs', default=11,
                           type=atypes.int_range(1),
                           help='number of training epochs per cross-val run')
        group.add_argument('--tb_dir', default=self.default_logdir,
                           help=('directory to store training logs for '
                                 'tensorboard.'))

        # network to train
        group = parser.add_argument_group(title="Network configuration")
        group.add_argument('-n', '--network', required=True,
                           metavar='NETWORK_NAME',
                           help='name of network module to use')
        group.add_argument('-m', '--model_file',
                           help='(optional) file with trained model of the '
                                'network')
        group.add_argument('--learning_rate', type=float,
                           help='learning rate to use')
        group.add_argument('--optimizer',
                           help='gradient descent optimizer to use')
        group.add_argument('--loss_fn',
                           help='loss function to use')

        # dataset input
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        dset_args = dargs.dataset_args(input_aliases=in_aliases)
        item_args = dargs.item_types_args()
        atype = dargs.arg_type.INPUT
        group = parser.add_argument_group(title="Input dataset")
        dset_args.add_dataset_arg_double(group, atype)
        item_args.add_item_type_args(group, atype)
        group.add_argument('--test_items_count', type=atypes.int_range(1),
                           help='number of dataset items to include in the '
                                'test set. Overrides test_items_fraction.')
        group.add_argument('--test_items_fraction', type=float, default=0.1,
                           help='number of dataset items to include in the '
                                'test set, expressed as a fraction.')

        self.parser = parser
        self.dset_args = dset_args
        self.item_args = item_args

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        logdir = args.tb_dir
        if logdir != self.default_logdir and not os.path.isdir(logdir):
            raise ValueError(('Invalid non-default log directory: {}'
                              .format(logdir)))

        args_dict = {}
        atype = dargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)
        args_dict['item_types'] = self.item_args.get_item_types(args, atype)
        name, srcdir = self.dset_args.get_dataset_double(args, atype)
        args_dict['name'], args_dict['srcdir'] = name, srcdir
        args_dict['test_items_count'] = args.test_items_count
        args_dict['test_items_fraction'] = args.test_items_fraction

        net_args = ('network', 'model_file', 'learning_rate', 'optimizer',
                    'loss_fn', )
        for attr in net_args:
            args_dict[attr] = getattr(args, attr)

        xval_args = ('num_crossvals', 'epochs', 'tb_dir', )
        for attr in xval_args:
            args_dict[attr] = getattr(args, attr)

        return args_dict
