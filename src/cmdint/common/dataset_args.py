import enum

import dataset.constants as cons


class arg_type(enum.Enum):
    INPUT = 'input'
    OUTPUT = 'output'


# loading or storing of datasets


class DatasetArgs:

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


class ItemTypeArgs:

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
        self.item_descriptions = {'raw': 'raw packets'}
        for k in cons.ALL_ITEM_TYPES[1:]:
            self.item_descriptions[k] = '{} packet projections'.format(k)

    def add_item_type_args(self, parser, atype, required_types={}, help={}):
        """
            Add dataset item type arguments to the given parser.

            This method adds as many arguments as there are item types listed
            in utils.dataset_utils. These arguments specify which dataset items
            to load from or store to secondary storage. Their use for either
            task is specified by the 'inputs' flag and their prefix.

            Parameters
            ----------
            :param parser:      the argparse parser to add the arguments to
            :type parser:       argparse.ArgumentParser
            :param atype:       flag specifying if these arguments are used
                                for loading or storing dataset items.
            :type atype:        cmdint.common_args.arg_type
            :param req_types:   (optional) dict of flags specifying which
                                dataset item types are mandatory.
            :type req_types:    typing.Mapping[str, bool]
            :param help:        (optional) dict of help descriptions per each
                                item type argument.
            :type help:         typing.Mapping[str, str]
        """
        if atype is arg_type.INPUT:
            prefix = self.input_prefix
            default_help = 'load {}'
        else:
            prefix = self.output_prefix
            default_help = 'store {}'
        desc = self.item_descriptions
        for k in cons.ALL_ITEM_TYPES:
            required = required_types.get(k, False)
            help_text = help.get(k, None) or default_help.format(desc[k])
            parser.add_argument('--{}_{}'.format(prefix, k), required=required,
                                action='store_true', help=help_text)
        return parser

    def get_item_types(self, args, atype):
        if atype is arg_type.INPUT:
            prefix = self.input_prefix
        else:
            prefix = self.output_prefix
        result = {}
        for k in cons.ALL_ITEM_TYPES:
            arg_name = '{}_{}'.format(prefix, k)
            result[k] = getattr(args, arg_name)
        return result
