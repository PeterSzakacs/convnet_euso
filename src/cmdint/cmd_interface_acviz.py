import os
import argparse

import cmdint.common.dataset_args as dargs
import cmdint.common.network_args as net_args


class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Visualize hidden layer activations of model using "
                        "given dataset")
        parser.add_argument('logdir',
                            help=('Directory to output visualized activation '
                                  'images to.'))

        # trained neural network model settings
        group = parser.add_argument_group('Trained model settings')
        net_args.add_network_arg(group, short_alias='n')
        net_args.add_model_file_arg(group, short_alias='m', required=True)

        # input dataset settings
        group = parser.add_argument_group(title="Input dataset")
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        dset_args = dargs.dataset_args(input_aliases=in_aliases)
        dset_args.add_dataset_arg_double(group, dargs.arg_type.INPUT,
                                         required=True,
                                         dir_default=os.path.curdir)
        item_args = dargs.item_types_args()
        item_args.add_item_type_args(group, dargs.arg_type.INPUT)
        group.add_argument('--start_item', default=0, type=int,
                           help=('index of first dataset item to use.'))
        group.add_argument('--stop_item', default=None, type=int,
                           help=('index of the dataset item after the last '
                                 'item to use.'))

        # misc
        parser.add_argument('--usecpu', action='store_true',
                            help=('Use host CPU instead of the CUDA device. '
                                  'On systems without a dedicated CUDA device '
                                  'and no CUDA-enabled version of tensorflow '
                                  'installed, this flag has no effect.'))
        self.dset_args = dset_args
        self.item_args = item_args
        self.parser = parser

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        logdir = args.logdir
        if not os.path.isdir(logdir):
            raise Exception('Invalid logdir: {}'.format(logdir))
        atype = dargs.arg_type.INPUT
        self.item_args.check_item_type_args(args, atype)
        name, srcdir = self.dset_args.get_dataset_double(args, atype)

        args_dict = {}
        args_dict['network'] = args.network
        args_dict['model_file'] = args.model_file
        args_dict['usecpu'] = args.usecpu
        args_dict['logdir'] = logdir
        args_dict['name'], args_dict['srcdir'] = name, srcdir
        args_dict['item_types'] = self.item_args.get_item_types(args, atype)
        args_dict['items_slice'] = slice(args.start_item, args.stop_item)

        return args_dict
