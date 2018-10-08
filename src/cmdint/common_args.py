import utils.packets.packet_utils as pack
import utils.dataset_utils as ds

# packet dimensions (directly required to create a packet template)

def add_packet_args(parser, required=True):
    parser.add_argument('--packet_dims', required=required, type=int, nargs=5, metavar=('NUM_GTU', 'HEIGHT', 'WIDTH', 'EC_HEIGHT', 'EC_WIDTH'),
                        help=('Dimensions of packets from which individual data items are created. Width and height'
                            ' of individual frames in a packet must be evenly divisible by EC width or height respectively'))

def packet_args_to_packet_template(args):
    n_gtu, f_h, f_w, ec_h, ec_w = args.packet_dims[0:5]
    return pack.packet_template(ec_w, ec_h, f_w, f_h, n_gtu)

def packet_args_to_string(args):
    n_gtu, f_h, f_w, ec_h, ec_w = args.packet_dims[0:5]
    n_data = args.num_data
    lam = args.bg_lambda
    bec_min, bec_max = args.bad_ECs[0:2]
    return 'pack_{}_{}_{}_{}_{}_num_{}_bad_ecs_{}-{}_lam_{}'.format(n_gtu, f_h, f_w, ec_h, ec_w, n_data, bec_min, bec_max, lam)

# type of packet data input to load from or output to store to files (since a neural network can use
# either only one packet projection type or multiple projections or even raw packets)

def add_input_type_dataset_args(parser, raw_required=False, yx_required=False, gtux_required=False, gtuy_required=False):
    parser.add_argument('--input_raw_packets', action='store_true', required=raw_required,
                            help=('Load dataset file containing raw packets'))
    parser.add_argument('--input_yx_proj', action='store_true', required=yx_required,
                            help=('Load dataset file containing YX projections of packets'))
    parser.add_argument('--input_gtux_proj', action='store_true', required=gtux_required,
                            help=('Load dataset file containing GTUX projections of packets'))
    parser.add_argument('--input_gtuy_proj', action='store_true', required=gtuy_required,
                            help=('Load dataset file containing GTUY projections of packets'))

def add_output_type_dataset_args(parser, raw_required=False, yx_required=False, gtux_required=False, gtuy_required=False):
    parser.add_argument('--create_raw_packets', action='store_true', required=raw_required,
                            help=('Create output file containing raw packets'))
    parser.add_argument('--create_yx_proj', action='store_true', required=yx_required,
                            help=('Create output file containing YX projections of packets'))
    parser.add_argument('--create_gtux_proj', action='store_true', required=gtux_required,
                            help=('Create output file containing GTUX projections of packets'))
    parser.add_argument('--create_gtuy_proj', action='store_true', required=gtuy_required,
                            help=('Create output file containing GTUY projections of packets'))

def check_input_type_dataset_args(args):
    raw, yx = args.input_raw_packets, args.input_yx_proj
    gtux, gtuy = args.input_gtux_proj, args.input_gtuy_proj
    no_input = not (raw or yx or gtux or gtuy)
    if (no_input):
        raise Exception('Please select at least one input type to load dataset packet data from (raw, yx, gtux, gtuy)')

def check_output_type_dataset_args(args):
    raw, yx = args.create_raw_packets, args.create_yx_proj
    gtux, gtuy = args.create_gtux_proj, args.create_gtuy_proj
    no_output = not (raw or yx or gtux or gtuy)
    if (no_output):
        raise Exception('Please select at least one output type to store dataset packet data (raw, yx, gtux, gtuy)')

def input_type_dataset_args_to_filenames(args, common_filename_part):
    raw, yx = args.input_raw_packets, args.input_yx_proj
    gtux, gtuy = args.input_gtux_proj, args.input_gtuy_proj
    return _dataset_file_types_to_filenames(common_filename_part, raw, yx, gtux, gtuy)

def output_type_dataset_args_to_filenames(args, common_filename_part):
    raw, yx = args.create_raw_packets, args.create_yx_proj
    gtux, gtuy = args.create_gtux_proj, args.create_gtuy_proj
    return _dataset_file_types_to_filenames(common_filename_part, raw, yx, gtux, gtuy)

def input_type_dataset_args_to_helper(args):
    raw, yx = args.input_raw_packets, args.input_yx_proj
    gtux, gtuy = args.input_gtux_proj, args.input_gtuy_proj
    return ds.numpy_dataset_helper(output_raw=raw, output_yx=yx, output_gtux=gtux, output_gtuy=gtuy)

def output_type_dataset_args_to_helper(args):
    raw, yx = args.create_raw_packets, args.create_yx_proj
    gtux, gtuy = args.create_gtux_proj, args.create_gtuy_proj
    return ds.numpy_dataset_helper(output_raw=raw, output_yx=yx, output_gtux=gtux, output_gtuy=gtuy)

def _dataset_file_types_to_filenames(common_filename_part, raw, yx, gtux, gtuy):
    data_types = (raw, yx, gtux, gtuy)
    file_tags = ('raw', 'yx', 'gtux', 'gtuy')
    filenames = tuple('{}_{}.npy'.format(common_filename_part, file_tags[idx])
                      for idx in range(4) if data_types[idx] == True)
    targetsfile = '{}_targets.npy'.format(common_filename_part)
    return filenames, targetsfile