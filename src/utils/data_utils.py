import math
import itertools
import random as rand

import numpy as np

# generator functions for shower line values

class default_vals_generator():
    
    def __init__(self, maximum, duration):
        self.reset(maximum, duration)
        
    def reset(self, maximum, duration):
        self.duration, self.maximum = duration, maximum
        self.iteration = 0
        self.maxinv = 1/maximum

    def __iter__(self):
        return self

    def __next__(self):
        if (self.iteration < self.duration):
            self.iteration += 1
            return round(-self.maxinv * pow(self.iteration - 2, 2) + self.maximum)
        else:
            raise StopIteration()

class flat_vals_generator():
    
    def __init__(self, maximum, duration):
        self.reset(maximum, duration)
        
    def reset(self, maximum, duration):
        self.iteration = 0
        self.duration = duration
        self.maximum = maximum

    def __iter__(self):
        return self

    def __next__(self):
        if (self.iteration < self.duration):
            self.iteration += 1
            return self.maximum
        else:
            raise StopIteration()

# class to perform various operations on packet data (create packet projections, 
# zero-out EC cells, draw a simulated shower line with a generator function, ...)

class packet_manipulator():

    def __init__(self, packet_template, verify_against_template=True):
        self.template = packet_template
        self.set_packet_verification(verify_against_template)

    def _verify_packet(self, packet, start_idx=0, end_idx=None):
        if packet.shape != self.template.packet_shape:
            raise ValueError("Illegal packet dimensions {}, expected packet with shape {}"
                                .format(packet.shape, self.template.packet_shape))

    """Turn on or off checking if packets passed to this objects methods conform to the template with which this object was created.

    Turning verification off might give slightly better performance.
    
    Parameters
    ----------
    verify_against_tempalte :   bool
                                Flag specifying whether to check packets on every method call.
    """
    def set_packet_verification(self, verify_against_template):
        if verify_against_template:
            self._check_packet = lambda packet: self._verify_packet(packet)
        else:
            self._check_packet = lambda packet: None

    """Randomly select regions of a packet frame corresponding to the EC modules on the surface of a source detector
    (can be used to e.g. simulate malfunctioning EC units of a detector).

    The number of regions or ECs actually selected is the minimum of (max_ECs) or (all possible ECs minus those in excluded_ECs).

    Parameters
    ----------
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
    def select_random_ECs(self, max_ECs, excluded_ECs=[]):
        EC_w, EC_h, EC_n = self.template.EC_width, self.template.EC_height, self.template.num_EC
        indices = set(range(0, EC_n, 1)).difference(excluded_ECs)
        used_indices = set()
        X, Y = [], []
        stop = min(len(indices), max_ECs)
        for iteration in range(0, stop):
            index = rand.randrange(0, EC_n)
            # do not use an EC index that was explicitly excluded nor one already used
            while not index in indices or index in used_indices:
                index = rand.randrange(0, EC_n)
            EC_x, EC_y = self.template.ec_idx_to_ec_xy(index)
            x_start, y_start = EC_x*EC_w, EC_y*EC_h
            x_stop, y_stop = x_start + EC_w, y_start + EC_h
            # perhaps it is better to not store slices but the actual X and Y positions
            X.append(slice(x_start, x_stop)), Y.append(slice(y_start, y_stop))
            used_indices.add(index)
            indices.discard(index)
        return tuple(X), tuple(Y), tuple(used_indices)
    
    """Create a simulated shower line and return its coordinates (GTU, X, Y) and values in the form of tuples of ints

    The shower line is inclined under ang_rad and starts at a given GTU, X and Y coordinate 
    Vals_generator is used to generate the values for every coordinate tuple. 

    The values and coordinates are trimmed if necessary to conform spatially to a packet conforming 
    to self.template (to prevent going outside template.num_frames or frame width or height)

    The returned tuples of coordinates and values can be used directly to draw the line into the actual packet.

    Parameters
    ----------
    start :             tuple of int
                        start coordinates from which to start the shower line in the form (start_gtu, start_x, start_y).
    ang_rad :           float
                        angle in radians under which the shower line appears in the xy projection of the packet.
    vals_generator :    iterable returning int
                        iterable object returning the desired shower value at each ieration, MUST BE PRESET TO GENERATE CORRECT VALUES.
    
    Returns
    -------
    X :                 tuple of int
                        tuple, where a given element represents the X-axis coordinate 
                        for a particular shower value at the same index in Values.
    Y :                 tuple of int
                        tuple, where a given element represents the Y-axis coordinate 
                        for a particular shower value at the same index in Values.
    GTU :               tuple of int
                        tuple, where a given element represents the GTU-axis coordinate 
                        for a particular shower value at the same index in Values.
    Values:             tuple of int
                        tuple of shower values in ordered ordered with respect to time (GTU)
    """
    def draw_simulated_shower_line(self, start, ang_rad, vals_generator):
        delta_x, delta_y = math.cos(ang_rad), math.sin(ang_rad)
        start_gtu, start_x, start_y = start[0], start[1], start[2]
        num_frames, height, width = self.template.num_frames, self.template.frame_height, self.template.frame_width

        # generate shower values for every GTU from start_gtu to num_frames or until the shower generator finishes
        Values = [val for val in itertools.islice(vals_generator, num_frames - start_gtu)]
        GTU = range(start_gtu, start_gtu + len(Values))
        # get x, y coordinates for every gtu
        X = [start_x + math.floor(delta_x * idx) for idx in range(len(Values))]
        Y = [start_y + math.floor(delta_y * idx) for idx in range(len(Values))]
        # remove those that are outside the edge of a packet frame
        X = [x for x in X if x >= 0 and x < width]
        Y = [y for y in Y if y >= 0 and y < height]
        num_frames = min(len(X), len(Y))
        X, Y, GTU, Values = tuple(X[:num_frames]), tuple(Y[:num_frames]), tuple(GTU[:num_frames]), tuple(Values[:num_frames])
        return X, Y, GTU, Values

    # projections (TODO: probably should be moved into a different module or class, 
    # since packet_manipulator is otherwise independent of numpy)

    def create_x_y_projection(self, packet, start_idx=0, end_idx=None):
        self._check_packet(packet)
        return np.max(packet[start_idx:end_idx], axis=0)
    
    def create_x_gtu_projection(self, packet, start_idx=0, end_idx=None):
        self._check_packet(packet)
        return np.max(packet[start_idx:end_idx], axis=1)
    
    def create_y_gtu_projection(self, packet, start_idx=0, end_idx=None):
        self._check_packet(packet)
        return np.max(packet[start_idx:end_idx], axis=2)