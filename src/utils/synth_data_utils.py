import itertools
import math
import random as rand

import utils.common_utils as cutils
import utils.data_utils as dat

"""Create a simulated shower line and return its coordinates (GTU, X, Y) and values in the form of tuples of ints

The shower line is inclined under ang_rad and starts at a given GTU, X and Y coordinate
Vals_generator is used to generate the values for every coordinate tuple.

The values and coordinates are trimmed if necessary to conform spatially to a packet conforming
to self.template (to prevent going outside template.num_frames or frame width or height)

The returned tuples of coordinates and values can be used directly to draw the line into the actual packet.

Parameters
----------
start_coordinate :  tuple of int
                    start coordinates from which to start the shower line in the form (start_gtu, start_y, start_x).
ang_rad :           float.
                    angle in radians under which the shower line appears in the xy projection of the packet.
packet_template :   utils.packets.packet_utils.packet_template.
                    template of the packet in which this simulated shower occurs. This is to shave off any coordinates
                    that are outside the packet boundaries.
vals_generator :    iterable returning int
                    iterable object returning the desired shower value at each ieration. MUST BE PRESET TO GENERATE
                    THE CORRECT VALUES FOR EVERY GTU.

Returns
-------
GTU :               tuple of int
                    tuple, where a given element represents the GTU-axis coordinate
                    for a particular shower value at the same index in Values.
Y :                 tuple of int
                    tuple, where a given element represents the Y-axis coordinate
                    for a particular shower value at the same index in Values.
X :                 tuple of int
                    tuple, where a given element represents the X-axis coordinate
                    for a particular shower value at the same index in Values.
Values:             tuple of int
                    tuple of shower values in ordered ordered with respect to time (GTU)
"""
def create_simu_shower_line(yx_angle, start_coordinate, packet_template, values_generator):
    ang_rad = math.radians(yx_angle)
    delta_x, delta_y = math.cos(ang_rad), math.sin(ang_rad)

    start_gtu, start_y, start_x = start_coordinate[0:3]
    num_frames, height, width = packet_template.num_frames, packet_template.frame_height, packet_template.frame_width

    # generate shower values for every GTU from start_gtu to num_frames or until the shower generator finishes
    Values = [val for val in itertools.islice(values_generator, num_frames - start_gtu)]
    GTU = range(start_gtu, start_gtu + len(Values))
    # get x, y coordinates for every gtu
    X = [start_x + math.floor(delta_x * idx) for idx in range(len(Values))]
    Y = [start_y + math.floor(delta_y * idx) for idx in range(len(Values))]
    # remove those that are outside the edge of a packet frame
    X = [x for x in X if x >= 0 and x < width]
    Y = [y for y in Y if y >= 0 and y < height]
    num_frames = min(len(X), len(Y))
    X, Y, GTU, Values = tuple(X[:num_frames]), tuple(Y[:num_frames]), tuple(GTU[:num_frames]), tuple(Values[:num_frames])
    return GTU, Y, X, Values

def create_simu_shower_line_from_template(shower_template, yx_angle):
    start = shower_template.get_new_start_coordinate()
    shower_max = shower_template.get_new_shower_max()
    duration = shower_template.get_new_shower_duration()
    vals_generator = shower_template.values_generator
    vals_generator.reset(shower_max, duration)

    packet_template = shower_template.packet_template
    return create_simu_shower_line(yx_angle, start, packet_template, vals_generator)

"""Randomly select regions of a packet frame corresponding to the EC modules on the surface of a source detector
(can be used to e.g. simulate malfunctioning EC units of a detector).

The number of regions or ECs actually selected is the minimum of (max_ECs) or (all possible ECs minus those in excluded_ECs).

Parameters
----------
packet_template :   utils.packets.packet_utils.packet_template
                    template of packet data
max_errors :        int
                    maximum number of ECs to select.
excluded_ECs :      (list or tuple) of ints
                    EC indexes that should not be selected.

Returns
-------
X :                 tuple of slices
                    tuple, where a given element represents the X-axis slice for an EC whose EC index
                    is in EC_indices at the same index as the slice.
Y :                 tuple of slices
                    tuple, where a given element represents the Y-axis slice for an EC whose EC index
                    is in EC_indices at the same index as the slice.
EC_indices :        tuple of int
                    EC indexes of selected ECs.
"""
def select_random_ECs(packet_template, max_ECs, excluded_ECs=[]):
    EC_n = packet_template.num_EC
    indices = set(range(0, EC_n, 1)).difference(excluded_ECs)
    used_indices = set()
    X, Y = [], []
    stop = min(len(indices), max_ECs)
    for iteration in range(0, stop):
        index = rand.randrange(0, EC_n)
        # do not use an EC index that was explicitly excluded nor one already used
        while not index in indices or index in used_indices:
            index = rand.randrange(0, EC_n)
        x, y = packet_template.ec_idx_to_xy_slice(index)
        # perhaps it is better to not store slices but the actual X and Y positions
        X.append(x), Y.append(y)
        used_indices.add(index)
        indices.discard(index)
    return tuple(X), tuple(Y), tuple(used_indices)



class simulated_shower_template():

    def __init__(self, packet_template, shower_duration, shower_max,
                        start_gtu=None, start_y=None, start_x=None,
                        values_generator=dat.default_vals_generator(10, 10)):
        self._template = packet_template
        self.shower_duration = shower_duration
        self.shower_max = shower_max
        # set default value ranges for start coordinates if not provided
        # let start coordinates be at least a distance of 3/4 * duration from the edges of a packet
        limit = int(3*self.shower_duration[1]/4)
        self.start_gtu = (0, packet_template.num_frames - limit) if start_gtu == None else start_gtu
        self.start_y = (limit, packet_template.frame_height - limit) if start_y == None else start_y
        self.start_x = (limit, packet_template.frame_width - limit) if start_x == None else start_x
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