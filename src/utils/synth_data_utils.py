"""
Collection of tools for parsing and checking parameters of synthetically generated data 
(i.e. data with simulated backgrounds instead of a background recorded during experiments).
"""

import utils.packets.packet_utils as pack

def _test_num(value, lower_bound, err_msg):
    if value < lower_bound:
        raise ValueError(err_msg)

# arguments that influence packet shape, number of packets and packet backgrounds
class params_args:

    def __init__(self):
        self._packet_dims_metavar = ('NUM_GTU', 'HEIGHT', 'WIDTH', 'EC_HEIGHT', 'EC_WIDTH')
    
    def add_packet_cmd_args(self, parser):
        parser.add_argument('--packet_dims', required=True, type=int, nargs=5, metavar=self._packet_dims_metavar,
                            help=('Dimensions of packets from which individual data items are created. Width and height'
                                ' of individual frames in a packet must be evenly divisible by EC width or height respectively'))

    def add_other_cmd_args(self, parser, bad_ECs_default=None):
        bad_ec_def = '' if bad_ECs_default == None else' Default value range: {}.'.format(bad_ECs_default)

        parser.add_argument('--num_data', required=True, type=int,
                            help=('Number of data items (both noise and shower), corresponds to number of packets'))
        parser.add_argument('--bg_lambda', required=True, type=float,
                            help=('Average of background pixel values (lambda in Poisson distributions'))
        parser.add_argument('--bad_ECs', required=True, type=int, nargs=2, metavar=('MIN', 'MAX'),
                            help=('Number of malfunctioned EC modules in the data. The actual number of such ECs'
                                ' in any data item is from MIN to MAX, inclusive. If MIN == MAX, the number of bad ECs'
                                ' is an exact number, barring cases where keeping this requirement would knock out ECs'
                                ' containing shower pixels.{}'.format(bad_ec_def)))

    def check_cmd_args(self, args):
        # a packet template already checks its own dimensions and raises errors in its constructor
        template = self.args_to_packet_template(args)
        _test_num(args.num_data, 1, 'number of data items must be greater than 0')
        _test_num(args.bg_lambda, 0, 'background mean (lambda) must be greater than or equal to 0')
        _test_num(args.bad_ECs[0], 0, 'minimum number of bad ECs cannot be a negative value')
        _test_num(args.bad_ECs[1], args.bad_ECs[0], 'maximum number of bad ECs cannot be less than the minimum value')
        _test_num(template.num_EC, args.bad_ECs[1], 'maximum number of bad ECs cannot exceed the total number of ECs per frame {}'.format(template.num_EC))

    def args_to_packet_template(self, args):
        n_gtu, f_h, f_w, ec_h, ec_w = args.packet_dims[0:5]
        return pack.packet_template(ec_w, ec_h, f_w, f_h, n_gtu)

    def args_to_string(self, args):
        n_gtu, f_h, f_w, ec_h, ec_w = args.packet_dims[0:5]
        n_data = args.num_data
        lam = args.bg_lambda
        bec_min, bec_max = args.bad_ECs[0:2]
        return 'pack_{}_{}_{}_{}_{}_num_{}_bad_ecs_{}-{}_lam_{}'.format(n_gtu, f_h, f_w, ec_h, ec_w, n_data, bec_min, bec_max, lam)

# arguments directly affecting simulated shower properties
class shower_args:

    def add_cmd_args(self, parser, duration_default=None, bg_diff_default=None, 
                    start_gtu_default=None, start_x_default=None, start_y_default=None):
        dur_def = '' if duration_default == None else ' Default value range: {}.'.format(duration_default)
        bg_diff_def = '' if bg_diff_default == None else ' Default value range: {}.'.format(bg_diff_default)
        start_gtu_def = '' if start_gtu_default == None else ' Default value range: {}.'.format(start_gtu_default)
        start_x_def = '' if start_x_default == None else ' Default value range: {}.'.format(start_x_default)
        start_y_def = '' if start_y_default == None else ' Default value range: {}.'.format(start_y_default)

        parser.add_argument('--bg_diff', metavar=('MIN', 'MAX'), nargs=2, type=int, required=True,
                                help=('Relative difference between pixel values of shower track and background. This is'
                                    ' a range of values from MIN to MAX, inclusive. If MIN == MAX, for all packets'
                                    ' the shower line has the same peak potential intensity.{}'.format(dur_def)))
        parser.add_argument('--duration', metavar=('MIN', 'MAX'), nargs=2, type=int, required=True,
                                help=('Duration of shower tracks in number of GTU or frames containing shower pixels.'
                                    ' The actual duration of a shower for any data item is from MIN to MAX, inclusive.'
                                    ' If MIN == MAX, the duration is always the same.{}'.format(dur_def)))
        parser.add_argument('--start_gtu', metavar=('MIN', 'MAX'), nargs=2, type=int,
                                help=('First GTU containing shower pixels. This is a range of GTUs from MIN to MAX, inclusive,'
                                    ' where a simulated shower line begins. If MIN == MAX, for all packets the shower line'
                                    ' starts at the same GTU in a packet.{}'.format(start_gtu_def)))
        parser.add_argument('--start_y', metavar=('MIN', 'MAX'), nargs=2, type=int,
                                help=('The y coordinate of a packet frame at which a shower line begins. This is a range'
                                    ' of y coordinate values from MIN to MAX, inclusive, where a shower line can begin.'
                                    ' If MIN == MAX, all packets have shower lines starting at the same y coordinate.{}'
                                    .format(start_y_def)))
        parser.add_argument('--start_x', metavar=('MIN', 'MAX'), nargs=2, type=int,
                                help=('The x coordinate of a packet frame at which a shower line begins. This is a range'
                                    ' of x coordinate values from MIN to MAX, inclusive, where a shower line can begin.'
                                    ' If MIN == MAX, all packets have shower lines starting at the same x coordinate.{}'
                                    .format(start_x_def)))

    def check_cmd_args(self, args, template):
        dims_and_limits = { 'start_gtu' : [template.num_frames, 'number of frames in a packet'], 
                            'start_x'   : [template.frame_width, 'width of a packet frame'], 
                            'start_y'   : [template.frame_height, 'height of a packet frame']}

        for key in dims_and_limits:
            val = getattr(args, key)
            if val != None:
                # MIN cannot be less than zero
                _test_num(val[0], 0, 'Lower bound for shower property {} must be greater than or eual to 0'.format(key))
                # MIN cannot be greater than MAX
                _test_num(val[1], val[0], 'Upper bound of shower property {} must be greater than or equal to its lower bound'.format(key))
                limit, msg = dims_and_limits[key][0:2]
                # MAX cannot exceed its respective packet dimension
                _test_num(limit, val[1], 'Upper bound of {} cannot be larger or the same as the {} ({})'.format(key, msg, limit))
        for key in ['bg_diff', 'duration']:
            val = getattr(args, key)
            _test_num(val[0], 1, 'Lower bound for shower property {} must be greater than or eual to 1'.format(key))
            # MIN cannot be greater than MAX
            _test_num(val[1], val[0], 'Upper bound of shower property {} must be greater than or equal to its lower bound'.format(key))
        if args.duration[1] >= template.num_frames:
            raise ValueError('Upper bound of shower duration cannot be larger than the number of frames in a packet')

    def args_to_dict(self, args):
        return {'start_gtu': args.start_gtu, 'start_x': args.start_x, 'start_y': args.start_y, 
                'bg_diff': args.bg_diff, 'duration': args.duration}

    def args_to_string(self, args):
        sx, sy, sgtu = args.start_x, args.start_y, args.start_gtu
        diff, dur = args.bg_diff, args.duration
        return 'shower_gtu_{}-{}_y_{}-{}_x_{}-{}_duration_{}-{}_bgdiff_{}-{}'.format(sgtu[0], sgtu[1], sy[0], sy[1], sx[0], sx[1], 
                                                                dur[0], dur[1], diff[0], diff[1])