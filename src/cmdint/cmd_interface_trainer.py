import os
import argparse

import cmdint.common.argparse_types as atypes
import cmdint.common.dataset_args as dargs
import cmdint.common.network_args as net_args
import net.constants as net_cons

class cmd_interface():

    def __init__(self):
        self.default_logdir = net_cons.DEFAULT_TRAIN_LOGDIR
        parser = argparse.ArgumentParser(
            description="Train network using provided dataset")

        # dataset input
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        dset_args = dargs.DatasetArgs(input_aliases=in_aliases)
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

        # network to train
        group = parser.add_argument_group(title="Network configuration")
        net_args.add_network_arg(group, short_alias='n')
        net_args.add_model_file_arg(group, short_alias='m')
        group.add_argument('--tb_dir', default=self.default_logdir,
                           help=('directory to store training logs for '
                                 'tensorboard.'))
        group.add_argument('--save', action='store_true',
                           help=('save the model after training. Model files '
                                 'are saved under tb_dir as net.network_name/'
                                 'net.network_name.tflearn.*'))

        # training settings
        group = parser.add_argument_group(title="Training parameters")
        net_args.add_training_settings_args(
            group, num_epochs={'required': False, 'default': 11,
                               'short_alias': 'e'})

        self.parser = parser
        self.dset_args = dset_args
        self.item_args = item_args

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        logdir = args.tb_dir
        if logdir != self.default_logdir and not os.path.isdir(logdir):
            raise ValueError(('Invalid non-default logging directory: {}'
                              .format(logdir)))

        args_dict = {}
        args_dict['tb_dir'], args_dict['save'] = logdir, args.save

        atype = dargs.arg_type.INPUT
        name, srcdir = self.dset_args.get_dataset_double(args, atype)
        args_dict['name'], args_dict['srcdir'] = name, srcdir
        args_dict['item_types'] = self.item_args.get_item_types(args, atype)
        args_dict['test_items_count'] = args.test_items_count
        args_dict['test_items_fraction'] = args.test_items_fraction
        args_dict['split_mode'] = args.split_mode

        network_args = ('network', 'model_file', )
        for attr in network_args:
            args_dict[attr] = getattr(args, attr)

        for key in net_args.TRAIN_SETTINGS_ARGS.keys():
            args_dict[key] = getattr(args, key)

        args_dict['tb_dir'], args_dict['save'] = logdir, args.save

        return args_dict
