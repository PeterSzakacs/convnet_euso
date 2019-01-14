import enum

import cmdint.argparse_types as atypes
import utils.data_templates as templates
import utils.dataset_utils as ds
import utils.metadata_utils as meta


class arg_type(enum.Enum):
    INPUT = 'input'
    OUTPUT = 'output'


# packet dimensions (directly required to create a packet template)


class packet_args:

    def __init__(self, long_alias='packet_dims'):
        self.long_alias = long_alias
        self.metavar = ('NUM_GTU', 'HEIGHT', 'WIDTH', 'EC_HEIGHT', 'EC_WIDTH')
        self.helpstr = ('Dimensions of packet data. Width and height must be'
                        ' evenly divisible by EC width or height respectively')

    def add_packet_arg(self, parser, short_alias=None, required=True):
        """
            Add argument for packet dimensions to the given parser

            Parameters
            ----------
            :param parser:  the argparse parser to add this argument to
            :type parser:   argparse.ArgumentParser
        """
        aliases = []
        if short_alias is not None:
            aliases.append('-{}'.format(short_alias))
        aliases.append('--{}'.format(self.long_alias))
        parser.add_argument(*aliases, metavar=self.metavar, nargs=5,
                            type=atypes.int_range(1), required=required,
                            help=self.helpstr)
        return parser

    def packet_arg_to_template(self, args):
        n_gtu, f_h, f_w, ec_h, ec_w = getattr(args, self.long_alias, None)[0:5]
        return templates.packet_template(ec_w, ec_h, f_w, f_h, n_gtu)

    def packet_arg_to_string(self, args):
        n_gtu, f_h, f_w, ec_h, ec_w = getattr(args, self.long_alias, None)[0:5]
        return 'pack_{}_{}_{}_{}_{}'.format(n_gtu, f_h, f_w, ec_h, ec_w)


# loading or storing of datasets


class dataset_args:

    input_arg_aliases = {
        'dataset name': 'input_name', 'dataset directory': 'srcdir',
        'dataset': 'input_dset'
    }
    output_arg_aliases = {
        'dataset name': 'output_name', 'dataset directory': 'outdir',
        'dataset': 'output_dset'
    }

    def __init__(self, input_aliases={}, output_aliases={}):
        self._in_alss, self._out_alss = {}, {}
        for arg_name, input in self.input_arg_aliases.items():
            output = self.output_arg_aliases[arg_name]
            # use user provided or default value as the arguments long form
            in_alias = input_aliases.get(arg_name, input)
            out_alias = output_aliases.get(arg_name, output)
            if in_alias == out_alias:
                raise Exception(("Error: the name of the input {} argument can"
                                 " not be the same as that of the equivalent"
                                 " output argument").format(arg_name))
            self._in_alss[arg_name] = in_alias
            self._out_alss[arg_name] = out_alias

    def add_dataset_arg_double(self, parser, atype, required=True,
                               name_short_alias=None, dir_short_alias=None,
                               name_default=None, dir_default=None):
        """
            Add dataset identification arguments to the given parser as a set
            of 2 arguments.

            This argument combination specifies which dataset to load from or
            store to secondary storage. Their use for either task is specified
            by the 'inputs' flag and if a command-line tool needs to do both
            dataset input and output, the long-form names of the corresponding
            pairs of arguments.

            Parameters
            ----------
            :param parser:              the parser to add the arguments to
            :type parser:               argparse.ArgumentParser
            :param atype:               flag specifying if these arguments are
                                        to be used for loading or saving a
                                        dataset.
            :type atype:                cmdint.common_args.arg_type
            :param required:            flag specifying if these arguments are
                                        mandatory.
            :type required:             bool
            :param name_short_alias:    optional short-form name of the dataset
                                        name argument
            :type name_short_alias:     str
            :param dir_short_alias:     optional short-form name of the dataset
                                        directory argument
            :type dir_short_alias:      str
            :param name_default:        optional default value for the dataset
                                        name argument
            :type name_default:         str
            :param dir_default:         optional default value for the dataset
                                        directory argument
            :type dir_default:          str
        """
        n_alss, d_alss = [], []
        if name_short_alias is not None:
            n_alss.append('-{}'.format(name_short_alias))
        if dir_short_alias is not None:
            d_alss.append('-{}'.format(dir_short_alias))
        if atype is arg_type.INPUT:
            n_alss.append('--{}'.format(self._in_alss['dataset name']))
            d_alss.append('--{}'.format(self._in_alss['dataset directory']))
        else:
            n_alss.append('--{}'.format(self._out_alss['dataset name']))
            d_alss.append('--{}'.format(self._out_alss['dataset directory']))
        parser.add_argument(*n_alss, required=required, default=name_default,
                            help='{} dataset name'.format(atype.value))
        parser.add_argument(*d_alss, required=required, default=dir_default,
                            help='{} dataset directory'.format(atype.value))
        return parser

    def add_dataset_arg_single(self, parser, atype, required=True,
                               multiple=False, short_alias=None,
                               input_metavars=('NAME', 'SRCDIR'),
                               output_metavars=('NAME', 'OUTDIR')):
        """
            Add dataset identification argument to the given parser as a single
            argument taking 2 values.

            This argument specifies which dataset to load from or store to
            secondary storage. Its use for either task is specified by the
            'inputs' flag and if both dataset input and output are required,
            the long-form name of this argument. This particular form of the
            argument is useful for example when working with multiple input
            and/or output datasets.

            The argument values are: the name of the dataset and the directory
            for all its files.

            Parameters
            ----------
            :param parser:          the argparse parser to add the argument to
            :type parser:           argparse.ArgumentParser
            :param atype:           flag specifying if this argument is used
                                    for loading or saving a dataset.
            :type atype:            cmdint.common_args.arg_type
            :param required:        flag specifying if this arguments is
                                    mandatory.
            :type required:         bool
            :param multiple:        flag specifying if this argument can be
                                    used multiple times on the command line
                                    (argparse append action)
            :type multiple:         bool
            :param short_alias:     optional short-form name of the argument
            :type short_alias:      str
            :param input_metavars:  optional tuple of metavars for the input
                                    form of this argument
            :type input_metavars:   (str, str)
            :param output_metavars: optional tuple of metavars for the output
                                    form of this argument
            :type output_metavars:  (str, str)
        """
        aliases = []
        if short_alias is not None:
            aliases.append('-{}'.format(short_alias))
        if atype is arg_type.INPUT:
            aliases.append('--{}'.format(self._in_alss['dataset']))
            metavars = input_metavars
        else:
            aliases.append('--{}'.format(self._out_alss['dataset']))
            metavars = output_metavars
        help_text = '{} dataset'.format(atype.value)
        if multiple:
            action = 'append'
            help_text += '(s)'
        else:
            action = 'store'
        parser.add_argument(*aliases, required=required, action=action,
                            nargs=2, metavar=metavars, help=help_text)
        return parser

    def get_dataset_single(self, args, atype):
        if atype is arg_type.INPUT:
            arg_name = self._in_alss['dataset']
        else:
            arg_name = self._out_alss['dataset']
        return getattr(args, arg_name)

    def get_dataset_double(self, args, atype):
        if atype is arg_type.INPUT:
            arg_n_name = self._in_alss['dataset name']
            arg_d_name = self._in_alss['dataset directory']
        else:
            arg_n_name = self._out_alss['dataset name']
            arg_d_name = self._out_alss['dataset directory']
        return getattr(args, arg_n_name), getattr(args, arg_d_name)


# loading or storing of dataset items


class item_types_args:

    def __init__(self, in_item_prefix='load', out_item_prefix='store'):
        if in_item_prefix == out_item_prefix:
            raise Exception(("Error: the common prefix of the input dataset"
                             " 'item types' argument can not be the same as"
                             " that of the output argument"))
        if in_item_prefix is None:
            raise Exception('Error: input dataset item type prefix is unset')
        if out_item_prefix is None:
            raise Exception('Error: output dataset item type prefix is unset')
        self.input_prefix = in_item_prefix
        self.output_prefix = out_item_prefix
        self._item_desc = {'raw': 'raw packets'}
        for k in ds.ALL_ITEM_TYPES[1:]:
            self._item_desc[k] = '{} packet projections'.format(k)

    def add_item_type_args(self, parser, atype, required_types={
                                k: False for k in ds.ALL_ITEM_TYPES}):
        """
            Add dataset item type arguments to the given parser.

            This method adds as many arguments as there are item types listed
            in utils.dataset_utils. These arguments specify which dataset items
            to load from or store to secondary storage. Their use for either
            task is specified by the 'inputs' flag and their prefix.

            Parameters
            ----------
            :param parser:          the argparse parser to add the arguments to
            :type parser:           argparse.ArgumentParser
            :param atype:           flag specifying if these arguments are used
                                    for loading or storing dataset items.
            :type atype:            cmdint.common_args.arg_type
            :param required_types:  dict of flags specifying which dataset item
                                    types are mandatory.
            :type required_types:   {str:bool}
        """
        if atype is arg_type.INPUT:
            prefix = self.input_prefix
            h_text = 'load {}'
        else:
            prefix = self.output_prefix
            h_text = 'store {}'
        for k in ds.ALL_ITEM_TYPES:
            desc = self._item_desc[k]
            required = required_types.get(k, False)
            parser.add_argument('--{}_{}'.format(prefix, k), required=required,
                                action='store_true', help=h_text.format(desc))
        return parser

    def check_item_type_args(self, args, atype):
        if atype is arg_type.INPUT:
            prefix = self.input_prefix
        else:
            prefix = self.output_prefix
        types_selected = False
        for k in ds.ALL_ITEM_TYPES:
            arg_name = '{}_{}'.format(prefix, k)
            types_selected = types_selected or getattr(args, arg_name)
        if not types_selected:
            raise Exception('Please select at least one item type: {}'.format(
                ds.ALL_ITEM_TYPES))

    def get_item_types(self, args, atype):
        if atype is arg_type.INPUT:
            prefix = self.input_prefix
        else:
            prefix = self.output_prefix
        result = {}
        for k in ds.ALL_ITEM_TYPES:
            arg_name = '{}_{}'.format(prefix, k)
            result[k] = getattr(args, arg_name)
        return result


# metadata field order


class metafield_order_arg:

    default_aliases = {k: k for k in meta.METADATA_TYPES.keys()}

    def __init__(self, order_arg_aliases={}):
        self._aliases = {}
        for meta_type in meta.METADATA_TYPES.keys():
            self._aliases[meta_type] = (order_arg_aliases.get(meta_type, None)
                                        or self.default_aliases[meta_type])

    def add_metafields_order_arg(self, parser, create_group=True,
                                 group_title='Metadata field order'):
        parser_or_group = (parser if not create_group
                           else parser.add_argument_group(title=group_title))
        for meta_type in meta.METADATA_TYPES.keys():
            alias = self._aliases[meta_type]
            parser_or_group.add_argument('--{}'.format(alias),
                                         action='store_const', const=meta_type,
                                         help=('{} metadata fields order'
                                               .format(meta_type)))
        return parser_or_group

    def get_metafields_order(self, args, none_selected_ok=False):
        meta_type = None
        num_orders = 0
        for alias in self._aliases.values():
            this_type = getattr(args, alias, None)
            if this_type is not None:
                meta_type = this_type
                num_orders += 1
        if num_orders > 1:
            raise Exception('Cannot select multiple metadata field orders')
        if meta_type is None:
            if none_selected_ok:
                return None
            else:
                raise Exception('No metafield order specified')
        else:
            return meta.METADATA_TYPES[meta_type]['field order']


# number range


def add_number_range_arg(parser, arg_name, short_alias=None, arg_desc=None,
                         arg_type=float, required=False, default=None,
                         metavar=('MIN', 'MAX')):
    aliases = []
    if short_alias is not None:
        aliases.append('-{}'.format(short_alias))
    aliases.append('--{}'.format(arg_name))
    if arg_desc is not None:
        help_txt = '{}. '.format(arg_desc)
    else:
        help_txt = 'Range of {} values. '.format(arg_name)
    help_txt += '{} == {} implies a constant value.'.format(*metavar)
    parser.add_argument(*aliases, type=arg_type, nargs=2, metavar=metavar,
                        required=required, default=default, help=help_txt)
    return parser
