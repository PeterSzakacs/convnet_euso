import random as rand

import utils.common_utils as cutils
import utils.shower_generators as gen

class packet_template():
    """Template for storing dimensions of packet data"""

    def __init__(self, EC_width, EC_height, frame_width, frame_height, frames_per_packet):
        arguments = dict(vars())
        del arguments['self']
        for key in arguments:
            if arguments[key] < 0:
                raise ValueError('Packet property {} cannot be a negative value, got: {}'.format(key, arguments[key]))
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

    #properties

    @property
    def packet_shape(self):
        return (self._num_frames, self._height, self._width)

    @property
    def frame_width(self):
        return self._width

    @property
    def frame_height(self):
        return self._height

    @property
    def EC_width(self):
        return self._EC_width

    @property
    def EC_height(self):
        return self._EC_height

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def num_cols(self):
        return self._num_cols

    @property
    def num_EC(self):
        return self._num_EC

    @property
    def num_frames(self):
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


class simulated_shower_template():
    """Template for storing parameters of generated showers"""

    def __init__(self, p_template, shower_duration, shower_max,
                        start_gtu=None, start_y=None, start_x=None,
                        values_generator=gen.default_vals_generator(10, 10)):
        if not isinstance(p_template, packet_template):
            raise TypeError("Required object of class packet_template as first argument, got {}".format(type(p_template)))
        self._template = p_template
        self.shower_duration = shower_duration
        self.shower_max = shower_max
        # set default value ranges for start coordinates if not provided
        # let start coordinates be at least a distance of 3/4 * duration from the edges of a packet
        limit = int(3*self.shower_duration[1]/4)
        self.start_gtu = (0, p_template.num_frames - limit) if start_gtu == None else start_gtu
        self.start_y = (limit, p_template.frame_height - limit) if start_y == None else start_y
        self.start_x = (limit, p_template.frame_width - limit) if start_x == None else start_x
        self.values_generator = values_generator
        # generator functions for shower parameter ranges
        static_lam = lambda min, max: min
        random_lam = lambda min, max: rand.randint(min, max)
        for prop_name in ['start_gtu', 'start_y', 'start_x', 'shower_max', 'shower_duration']:
            min, max = getattr(self, prop_name)
            setattr(self, '_{}_gen'.format(prop_name), (static_lam if min == max else random_lam))

    # shower properties

    @property
    def packet_template(self):
        return self._template

    @property
    def start_gtu(self):
        return self._start_gtu

    @start_gtu.setter
    def start_gtu(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'start_gtu')
        cutils.check_interval_tuple(vals, 'start_gtu', lower_limit=0, upper_limit=self._template.num_frames - 1)
        self._start_gtu = vals

    @property
    def start_y(self):
        return self._start_y

    @start_y.setter
    def start_y(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'start_y')
        cutils.check_interval_tuple(vals, 'start_y', lower_limit=0, upper_limit=self._template.frame_height - 1)
        self._start_y = vals

    @property
    def start_x(self):
        return self._start_x

    @start_x.setter
    def start_x(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'start_x')
        cutils.check_interval_tuple(vals, 'start_x', lower_limit=0, upper_limit=self._template.frame_width - 1)
        self._start_x = vals

    @property
    def shower_max(self):
        return self._max

    @shower_max.setter
    def shower_max(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'shower_max')
        cutils.check_interval_tuple(vals, 'shower_max', lower_limit=1)
        self._max = vals

    @property
    def shower_duration(self):
        return self._duration

    @shower_duration.setter
    def shower_duration(self, value):
        vals = cutils.check_and_convert_value_to_tuple(value, 'shower_duration')
        cutils.check_interval_tuple(vals, 'shower_duration', lower_limit=1, upper_limit=self._template.num_frames)
        self._duration = vals

    @property
    def values_generator(self):
        return self._vals_generator

    @values_generator.setter
    def values_generator(self, value):
        self._vals_generator = value

    def get_new_start_coordinate(self):
        start_gtu = self._start_gtu_gen(*(self._start_gtu))
        start_y = self._start_y_gen(*(self._start_y))
        start_x = self._start_x_gen(*(self._start_x))
        return (start_gtu, start_y, start_x)

    def get_new_shower_max(self):
        return self._shower_max_gen(*(self._max))

    def get_new_shower_duration(self):
        return self._shower_duration_gen(*(self._duration))


class synthetic_background_template():
    """Template for storing information about a synthetic (simulated) background"""

    def __init__(self, p_template, bg_lambda=(1, 1),
                 bad_ECs_range=(0, 0)):
        if not isinstance(p_template, packet_template):
            raise TypeError("Required object of class packet_template as first argument, got {}".format(type(p_template)))
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
            Tuple of 2 integers (MIN, MAX) representing the average background
            value of pixels in the data items.

            The actual average value is different from item to item but always
            from MIN to MAX inclusive. MIN == MAX impies a constant value for
            all items. MIN can not be less than 0.
        """
        return self._bg_lambda

    @bg_lambda_range.setter
    def bg_lambda_range(self, value):
        interval = cutils.check_and_convert_value_to_tuple(value,
                                                           'bg_lambda_range')
        cutils.check_interval_tuple(interval, 'bg_lambda_range', lower_limit=0)
        self._bg_lambda = interval
        lam_min, lam_max = interval[0:2]
        self._bg_lambda_gen = ((lambda: lam_min) if lam_min == lam_max else
                               (lambda: rand.uniform(lam_min, lam_max)))

    @property
    def bad_ECs_range(self):
        """
            Tuple of 2 integers (MIN, MAX) representing how many EC units
            within the data should haved simulated malfunctions.

            The effect of EC malfunction is simulated for half of all items in
            the dataset, where the actual number of these units per frame is
            from MIN to MAX inclusive. MIN == MAX impies a constant number of
            malfunctioned EC units per frame. MAX can not be more than the
            total number of ECs per frame. Note also that in case of packets
            with simulated showers, some EC cannot be malfunctioned. Therefore,
            even if this property were set to (num ECs, num ECs), data items
            with smulated showers shall never have every EC unit malfunctioned.
        """
        return self._bad_ECs

    @bad_ECs_range.setter
    def bad_ECs_range(self, value):
        interval = cutils.check_and_convert_value_to_tuple(value,
                                                           'bad_ECs_range')
        cutils.check_interval_tuple(interval, 'bad_ECs_range', 0,
                                    self._template.packet_template.num_EC)
        self._bad_ECs = interval
        EC_min, EC_max = interval[0:2]
        self._bad_ECs_gen = ((lambda: EC_min) if EC_min == EC_max else
                             (lambda: rand.randint(EC_min, EC_max)))

    def get_new_bg_lambda(self):
        return self._bg_lambda_gen()

    def get_new_bad_ECs(self):
        return self._bad_ECs_gen()