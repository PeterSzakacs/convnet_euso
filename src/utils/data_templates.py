import random as rand

import utils.common_utils as cutils
import utils.shower_generators as gen


class packet_template(cutils.CommonEqualityMixin):
    """Template for storing dimensions of packet data"""

    def __init__(self, EC_width, EC_height, frame_width, frame_height,
                 frames_per_packet):
        arguments = dict(vars())
        del arguments['self']
        for key in arguments:
            if arguments[key] < 0:
                raise ValueError(('Property {} cannot be a negative value,'
                                 ' got: {}').format(key, arguments[key]))
        if frame_width % EC_width != 0:
            raise ValueError('Frame width must be a multiple of EC width')
        if frame_height % EC_height != 0:
            raise ValueError('Frame height must be a multiple of EC height')

        self._EC_width, self._EC_height = EC_width, EC_height
        self._width, self._height = frame_width, frame_height
        self._num_rows = int(frame_height/EC_height)
        self._num_cols = int(frame_width/EC_width)
        self._num_EC = self._num_rows * self._num_cols
        self._num_frames = frames_per_packet

    # properties

    @property
    def packet_shape(self):
        """
            Shape of the packets expressed as a tuple of:
            (num_frames, frame_height, frame_width)
        """
        return (self._num_frames, self._height, self._width)

    @property
    def frame_width(self):
        """Number of pixels along the horizontal (X) axis of a packet frame"""
        return self._width

    @property
    def frame_height(self):
        """Number of pixels along the vertical (Y) axis of a packet frame"""
        return self._height

    @property
    def EC_width(self):
        """Number of pixels along the horizontal (X) axis of an EC unit"""
        return self._EC_width

    @property
    def EC_height(self):
        """Number of pixels along the vertical (Y) axis of an EC unit"""
        return self._EC_height

    @property
    def num_rows(self):
        """Number of ECs along the vertical (Y) axis of a packet frame"""
        return self._num_rows

    @property
    def num_cols(self):
        """Number of ECs along the horizontal (X) axis of a packet frame"""
        return self._num_cols

    @property
    def num_EC(self):
        """Number of ECs per packet frame"""
        return self._num_EC

    @property
    def num_frames(self):
        """Number of frames or GTU per packet"""
        return self._num_frames

    # unit conversions

    def x_to_ec_x(self, x):
        return int(x/self._EC_width)

    def y_to_ec_y(self, y):
        return int(y/self._EC_height)

    def xy_to_ec_idx(self, x, y):
        return self.ec_xy_to_ec_idx(self.x_to_ec_x(x), self.y_to_ec_y(y))

    def ec_xy_to_ec_idx(self, ec_x, ec_y):
        return ec_x + self._num_cols * ec_y

    def ec_idx_to_ec_xy(self, ec_idx):
        return ec_idx % self._num_cols, int(ec_idx / self._num_cols)

    def ec_idx_to_xy_slice(self, ec_idx):
        EC_w, EC_h = self._EC_width, self._EC_height
        EC_x, EC_y = self.ec_idx_to_ec_xy(ec_idx)
        x_start, y_start = EC_x*EC_w, EC_y*EC_h
        x_stop, y_stop = x_start + EC_w, y_start + EC_h
        return slice(x_start, x_stop), slice(y_start, y_stop)


class simulated_shower_template(cutils.CommonEqualityMixin):
    """Template for storing parameters of generated showers"""

    def __init__(self, p_template, shower_duration, shower_max,
                 start_gtu=None, start_y=None, start_x=None,
                 values_generator=None):
        if not isinstance(p_template, packet_template):
            raise TypeError(('Required object of type packet_template as first'
                            ' argument, got {}').format(type(p_template)))
        self._template = p_template
        self.shower_duration = shower_duration
        self.shower_max = shower_max
        # Set default value ranges for start coordinates and value generator,
        # if not provided
        # Let start coordinates be at least a distance of 3/4 * duration from
        # the edges of a packet
        limit = int(3*self.shower_duration[1]/4)
        self.start_gtu = start_gtu or (0, p_template.num_frames - limit)
        self.start_y = start_y or (limit, p_template.frame_height - limit)
        self.start_x = start_x or (limit, p_template.frame_width - limit)
        self.values_generator = (values_generator or
                                 gen.default_vals_generator(10, 10))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if not isinstance(other._vals_generator,
                              self._vals_generator.__class__):
                return False
            else:
                d1 = self.__dict__.copy()
                del d1['_vals_generator']
                d2 = other.__dict__.copy()
                del d2['_vals_generator']
                return d1 == d2
        else:
            return False

    # shower properties

    @property
    def packet_template(self):
        return self._template

    @property
    def start_gtu(self):
        """
            Tuple of 2 integers, MIN and MAX, representing the range of first
            GTUs (packet frames) usable for the first pixels of a shower track.

            The actual value is randomly generated
        """
        return self._start_gtu

    @start_gtu.setter
    def start_gtu(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'start_gtu')
        limits = (0, self._template.num_frames - 1)
        cutils.check_interval_tuple(vals, 'start_gtu', lower_limit=limits[0],
                                    upper_limit=limits[1])
        self._start_gtu = vals

    @property
    def start_y(self):
        """
            Tuple of 2 integers, MIN and MAX, representing the range of
            vertical (Y) coordinates usable for the first pixel of a
            shower track in the first frame (GTU).
        """
        return self._start_y

    @start_y.setter
    def start_y(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'start_y')
        limits = (0, self._template.frame_height - 1)
        cutils.check_interval_tuple(vals, 'start_y', lower_limit=limits[0],
                                    upper_limit=limits[1])
        self._start_y = vals

    @property
    def start_x(self):
        """
            Tuple of 2 integers, MIN and MAX, representing the range of
            horizontal (X) coordinates usable for the first pixel of a
            shower track in the first frame (GTU).
        """
        return self._start_x

    @start_x.setter
    def start_x(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'start_x')
        limits = (0, self._template.frame_width - 1)
        cutils.check_interval_tuple(vals, 'start_x', lower_limit=limits[0],
                                    upper_limit=limits[1])
        self._start_x = vals

    @property
    def shower_max(self):
        """
            Tuple of 2 integers, MIN and MAX, representing the range of peak
            shower intensities usable. Shower intensity is expressed relative
            to the previous pixel values at the given spots.
        """
        return self._max

    @shower_max.setter
    def shower_max(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'shower_max')
        cutils.check_interval_tuple(vals, 'shower_max', lower_limit=1)
        self._max = vals

    @property
    def shower_duration(self):
        """
            Tuple of 2 integers, MIN and MAX, representing the range of shower
            duration values usable. Duration of the shower is expressed as the
            number of consecutive frames (or GTUs) on which shower track pixels
            are located.
        """
        return self._duration

    @shower_duration.setter
    def shower_duration(self, value):
        vals = cutils.check_and_convert_value_to_tuple(
            value, 'shower_duration'
        )
        limits = (1, self._template.num_frames)
        cutils.check_interval_tuple(vals, 'shower_duration',
                                    lower_limit=limits[0],
                                    upper_limit=limits[1])
        self._duration = vals

    @property
    def values_generator(self):
        return self._vals_generator

    @values_generator.setter
    def values_generator(self, value):
        self._vals_generator = value

    def get_new_start_coordinate(self):
        """
            Generate a random new start coordinate for the shower and return it
            as a tuple of integers with the meaning: (GTU, Y, X)
        """
        start_gtu = rand.randint(*(self._start_gtu))
        start_y = rand.randint(*(self._start_y))
        start_x = rand.randint(*(self._start_x))
        return (start_gtu, start_y, start_x)

    def get_new_shower_max(self):
        return rand.randint(*(self._max))

    def get_new_shower_duration(self):
        return rand.randint(*(self._duration))


class synthetic_background_template(cutils.CommonEqualityMixin):
    """
        Template for storing information about a synthetic (simulated)
        background
    """

    def __init__(self, p_template, bg_lambda=(1.0, 1.0),
                 bad_ECs_range=(0, 0)):
        if not isinstance(p_template, packet_template):
            raise TypeError(('First parameter must be an object of type'
                            ' packet_template, instead got {}').format(
                            type(p_template)))
        self._template = p_template
        self.bg_lambda_range = bg_lambda
        self.bad_ECs_range = bad_ECs_range

    # background properties

    @property
    def packet_template(self):
        """
            Template for packet data.

            Constrains the maximum allowable number of bad EC units to at most
            template.num_ECs.
        """
        return self._template

    @property
    def bg_lambda_range(self):
        """
            Tuple of 2 floats (MIN, MAX) representing the range of average
            background pixel values (not counting any potential shower track
            pixels) on a per-packet basis.

            The actual average value per packet is always from MIN to MAX
            inclusive. MIN == MAX impies the same value for all packets. MIN
            can not be less than 0.
        """
        return self._bg_lambda

    @bg_lambda_range.setter
    def bg_lambda_range(self, value):
        interval = cutils.check_and_convert_value_to_tuple(
            value, 'bg_lambda_range'
        )
        cutils.check_interval_tuple(interval, 'bg_lambda_range', lower_limit=0)
        self._bg_lambda = interval

    @property
    def bad_ECs_range(self):
        """
            Tuple of 2 integers (MIN, MAX) representing how many EC units
            on a per-packet basis should haved simulated malfunctions.

            The actual value per packets is always from MIN to MAX inclusive.
            MIN == MAX impies a constant number of malfunctioned EC units per
            packet. MIN, cannot be less than 0 and MAX can not be more than
            the total number of EC units per packet frame.
        """
        return self._bad_ECs

    @bad_ECs_range.setter
    def bad_ECs_range(self, value):
        interval = cutils.check_and_convert_value_to_tuple(
            value, 'bad_ECs_range'
        )
        cutils.check_interval_tuple(interval, 'bad_ECs_range', 0,
                                    self._template.num_EC)
        self._bad_ECs = interval

    def get_new_bg_lambda(self):
        return rand.uniform(*(self._bg_lambda))

    def get_new_bad_ECs(self):
        return rand.randint(*(self._bad_ECs))
