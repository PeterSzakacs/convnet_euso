# classes and functions for manipulating data in packets of simulated frames

import math
import random as rand

# generator functions for shower line values

class default_vals_generator():
    
    def __init__(self, maximum, duration):
        self.reset(maximum, duration)
        
    def reset(self, maximum, duration):
        self.duration, self.nmaximum = duration, maximum
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

# Packet data manipulations

class packet_manipulator():

    def __init__(self, EC_width, EC_height, frame_width, frame_height, track_ECs=True):
        self._EC_width, self._EC_height = EC_width, EC_height
        self._width, self._height = frame_width, frame_height
        self._num_rows = int(frame_height/EC_height)
        self._num_cols = int(frame_width/EC_width)
        self._num_EC = self._num_rows * self._num_cols
        self.track_ECs = track_ECs

    #properties and setters

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
    def track_ECs(self):
        return self._track_ECs
    
    @track_ECs.setter
    def track_ECs(self, value):
        self._track_ECs = value
        if value:
            self._EC_tracker_fn = lambda x, y, EC_log: EC_log.append(self.xy_to_ec_idx(x, y))
        else:
            self._EC_tracker_fn = lambda x, y, EC_log: None

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
    
    # the actual magic

    def simu_EC_malfunction(self, packet, num_malfunctions_max, shower_EC_indexes=[]):
        indices = set(range(0, self._num_EC, 1)).difference(shower_EC_indexes)
        used_indices = set()
        stop = min(len(indices), num_malfunctions_max)
        for iteration in range(0, stop):
            index = rand.randrange(0, self._num_EC)
            # do not use an EC index where shower pixels are located nor one already used
            while not index in indices or index in used_indices:
                index = rand.randrange(0, self._num_EC)
            EC_x, EC_y = self.ec_idx_to_ec_xy(index)
            x_start, y_start = EC_x*self._EC_width, EC_y*self._EC_height
            x_stop, y_stop = x_start + self._EC_width, y_start + self._EC_height
            (packet[:, x_start:x_stop, y_start:y_stop]).fill(0)
            used_indices.add(index)
            indices.discard(index)
    
    def draw_simulated_shower_line(self, packet, start_x, start_y, ang_rad, vals_generator, start_gtu=2):
        delta_x = math.cos(ang_rad)
        delta_y = math.sin(ang_rad)
        frame_index, iteration_index = start_gtu, 0
        ECs_used = []
        for val in vals_generator:
            frame = packet[frame_index]
            offset_x = start_x + math.floor(delta_x * iteration_index)
            offset_y = start_y + math.floor(delta_y * iteration_index)
            at_edge = (offset_x < 0 or offset_x >= self._width or offset_y < 0 or offset_y >= self._height)
            if at_edge:
                break
            frame[offset_x][offset_y] += val
            ECs_used.append(self.xy_to_ec_idx(offset_x, offset_y))
            frame_index += 1
            iteration_index += 1
        return ECs_used
        