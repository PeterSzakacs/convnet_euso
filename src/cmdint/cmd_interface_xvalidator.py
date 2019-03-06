import os
import argparse

import net.constants as net_cons
import cmdint.common.argparse_types as atypes
import cmdint.common.dataset_args as dargs
import cmdint.common.network_args as net_args

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

        # network to train
        group = parser.add_argument_group(title="Network to use")
        net_args.add_network_arg(group, short_alias='n')
        net_args.add_model_file_arg(group, short_alias='m')

        # training_parameters
        group = parser.add_argument_group(title="Training parameters to use")
        net_args.add_training_settings_args(
            group, num_epochs={'required': False, 'default': 11,
                               'short_alias': 'e'})
        group.add_argument('--tb_dir', default=self.default_logdir,
                           help=('directory to store training logs for '
                                 'tensorboard.'))

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

        network_args = ('network', 'model_file', )
        for attr in network_args:
            args_dict[attr] = getattr(args, attr)

        for key in net_args.TRAIN_SETTINGS_ARGS.keys():
            args_dict[key] = getattr(args, key)

        xval_args = ('num_crossvals', 'tb_dir', )
        for attr in xval_args:
            args_dict[attr] = getattr(args, attr)

        return args_dict
