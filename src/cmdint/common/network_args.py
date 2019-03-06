import cmdint.common.argparse_types as atypes


DEFAULT_ARGPARSE_VALUES = {'short_alias': None, 'type': str, 'default': None,
                           'action': 'store',  'required': False, }


TRAIN_SETTINGS_ARGS = {
    'num_epochs': {'type': atypes.int_range(1), 'required': True,
                   'help': 'Number of training epochs', },
    'optimizer': {'help': 'Gradient descent optimizer to use', },
    'learning_rate': {'type': float, 'help': 'Learning rate to use', },
    'loss_fn': {'help': 'Loss function to use', },
    'batch_size': {'type': atypes.int_range(1),
                   'help': 'Batch size for training data', },
    'validation_batch_size': {'type': atypes.int_range(1),
                              'help': 'Batch size for test data', },
    'snapshot_step': {'type': atypes.int_range(1),
                      'help': 'Number of training steps between snapshots', },
}


def add_training_settings_args(parser, excluded_args=[], **settings):
    for key, val in TRAIN_SETTINGS_ARGS.items():
        if key in excluded_args:
            continue
        values = settings.get(key, val)
        arg_action = values.get('action', DEFAULT_ARGPARSE_VALUES['action'])
        arg_type = values.get('type', DEFAULT_ARGPARSE_VALUES['type'])
        arg_default = values.get('default', DEFAULT_ARGPARSE_VALUES['default'])
        arg_req = values.get('required', DEFAULT_ARGPARSE_VALUES['required'])
        arg_help = values.get('help', val['help'])
        short_alias = values.get('short_alias',
                                 DEFAULT_ARGPARSE_VALUES['short_alias'])
        aliases = []
        if short_alias is not None:
            aliases.append('-{}'.format(short_alias))
        aliases.append('--{}'.format(key))
        parser.add_argument(*aliases, action=arg_action, default=arg_default,
                            required=arg_req, type=arg_type, help=arg_help)

def add_network_arg(parser, arg_name='network', short_alias=None,
                    required=True, help='Name of network module/architecture'):
    aliases = []
    if short_alias is not None:
        aliases.append('-{}'.format(short_alias))
    aliases.append('--{}'.format(arg_name))
    parser.add_argument(*aliases, required=required, help=help)


def add_model_file_arg(parser, arg_name='model_file', short_alias=None,
                       required=False,
                       help='File with trained model of given architecture'):
    aliases = []
    if short_alias is not None:
        aliases.append('-{}'.format(short_alias))
    aliases.append('--{}'.format(arg_name))
    parser.add_argument(*aliases, required=required, help=help)
