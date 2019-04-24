import enum

import cmdint.common.argparse_types as atypes
import dataset.constants as cons
import utils.data_templates as templates


# packet dimensions (directly required to create a packet template)


class PacketArgs:

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
        return templates.PacketTemplate(ec_w, ec_h, f_w, f_h, n_gtu)

    def packet_arg_to_string(self, args):
        n_gtu, f_h, f_w, ec_h, ec_w = getattr(args, self.long_alias, None)[0:5]
        return 'pack_{}_{}_{}_{}_{}'.format(n_gtu, f_h, f_w, ec_h, ec_w)


# metadata field order


class MetafieldOrderArg:

    default_aliases = {k: k for k in cons.METADATA_TYPES.keys()}

    def __init__(self, order_arg_aliases={}):
        self._aliases = {}
        for meta_type in cons.METADATA_TYPES.keys():
            self._aliases[meta_type] = (order_arg_aliases.get(meta_type, None)
                                        or self.default_aliases[meta_type])

    def add_metafields_order_arg(self, parser, create_group=True,
                                 group_title='Metadata field order'):
        parser_or_group = (parser if not create_group
                           else parser.add_argument_group(title=group_title))
        for meta_type in cons.METADATA_TYPES.keys():
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
            return cons.METADATA_TYPES[meta_type]['field order']


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
