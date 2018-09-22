import math
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
            return -self.maxinv * pow(self.iteration - 2, 2) + self.maximum
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

    """Zero out regions of a packet as seen in the XY projection to simulate malfunctioning EC units of a detector.

    The number of regions or ECs actually affected is the minimum of (max_errors) or (all possible ECs minus those in excluded_ECs).

    Parameters
    ----------
    packet :            numpy.ndarraye
                        The packet to edit.
    max_errors :        int
                        maximum number of malfunctioning ECs to create.
    excluded_ECs :      (list or tuple) of ints
                        EC indexes that should not be zeroed out.
    """
    def simu_EC_malfunction(self, packet, max_errors, excluded_ECs=[]):
        self._check_packet(packet)
        EC_w, EC_h, EC_n = self.template.EC_width, self.template.EC_height, self.template.num_EC
        indices = set(range(0, EC_n, 1)).difference(excluded_ECs)
        used_indices = set()
        stop = min(len(indices), max_errors)
        for iteration in range(0, stop):
            index = rand.randrange(0, EC_n)
            # do not use an EC index where shower pixels are located nor one already used
            while not index in indices or index in used_indices:
                index = rand.randrange(0, EC_n)
            EC_x, EC_y = self.template.ec_idx_to_ec_xy(index)
            x_start, y_start = EC_x*EC_w, EC_y*EC_h
            x_stop, y_stop = x_start + EC_w, y_start + EC_h
            (packet[:, y_start:y_stop, x_start:x_stop]).fill(0)
            used_indices.add(index)
            indices.discard(index)
    
    """Draw a simulated shower line using vals_generator into a packet conforming to the passed packet template

    Parameters
    ----------
    packet :            numpy.ndarray
                        The packet to draw the shower line into.
    start :             tuple of int
                        start coordinates from which to start the shower line in the form (start_gtu, start_x, start_y).
    ang_rad :           float
                        angle in radians under which the shower line appears in the xy projection of the packet.
    vals_generator :    utils.shower_generators.*
                        iterable object returning the desired shower value at each ieration, MUST BE PRESET TO GENERATE CORRECT VALUES.
    
    Returns
    -------
    list of int
        A list of all ECs (as EC indexes) on which the simulated shower track was drawn.
    """
    def draw_simulated_shower_line(self, packet, start, ang_rad, vals_generator):
        self._check_packet(packet)
        start_gtu, start_x, start_y = start[0], start[1], start[2]
        width, height = self.template.frame_width, self.template.frame_height
        delta_x = math.cos(ang_rad)
        delta_y = math.sin(ang_rad)
        frame_index, iteration_index = start_gtu, 0
        ECs_used = []
        for val in vals_generator:
            frame = packet[frame_index]
            offset_x = start_x + math.floor(delta_x * iteration_index)
            offset_y = start_y + math.floor(delta_y * iteration_index)
            at_edge = (offset_x < 0 or offset_x >= width or offset_y < 0 or offset_y >= height)
            if at_edge:
                break
            frame[offset_y][offset_x] += val
            ECs_used.append(self.template.xy_to_ec_idx(offset_x, offset_y))
            frame_index += 1
            iteration_index += 1
        return ECs_used

    # projections

    def create_x_y_projection(self, packet, start_idx=0, end_idx=None):
        self._check_packet(packet)
        return np.max(packet[start_idx:end_idx], axis=0)
    
    def create_x_gtu_projection(self, packet, start_idx=0, end_idx=None):
        self._check_packet(packet)
        return np.max(packet[start_idx:end_idx], axis=1)
    
    def create_y_gtu_projection(self, packet, start_idx=0, end_idx=None):
        self._check_packet(packet)
        return np.max(packet[start_idx:end_idx], axis=2)