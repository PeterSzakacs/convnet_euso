import os
import argparse

from abc import ABC

class base_cmd_interface(ABC):

    def __init__(self):
        self._param_metavars = ('WIDTH', 'HEIGHT', 'NUM_FRAMES', 'LAMBDA', 'BG_DIFF', 'NUM_MERGED')

    # protected
    def _positive_int(self, value):
        val = 0
        try:
            val = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError('not an integer: ' + value)
        if val < 1:
            raise argparse.ArgumentTypeError('must be an integer greater than 0')
        return val

    # protected
    def _positive_float(self, value):
        val = 0
        try:
            val = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError('not a floating-point number: ' + value)
        if val <= 0:
            raise argparse.ArgumentTypeError('must be a floating point number greater than 0')
        return val

    # private
    def __check_params(self, directory, params):
        if not os.path.exists(directory):
           raise ValueError('directory for storing input and target files does not exist')

        p = params
        width, height, num_frames, lam, bg_diff, num_merged = p[0], p[1], p[2], p[3], p[4], p[5]
        if width < 28 or width > 48:
            raise ValueError('frame width must be >= 28 and <= 48')
        if height < 28 or height > 48:
            raise ValueError('frame height must be >= 28 and <= 48')
        if num_frames < 1:
            raise ValueError('number of frames must be greater than 0')
        if lam < 0:
            raise ValueError('mean background noise (lambda) must be a non-negative integer')
        if bg_diff < 1:
            raise ValueError('shower intensity relative to background must be greater than 0')
        if num_merged < 1:
            raise ValueError('number of bacckground frames to merge must be greater than 0')
        return width, height, num_frames, lam, bg_diff, num_merged

    def params_to_filenames(self, directory, params):
        width, height, num_frames, lam, bg_diff, num_merged = self.__check_params(directory, params)
        datafile   = os.path.join(directory, 'simu_data_x_{}_{}_{}_lam_{}_diff_{}_merge_{}'
                        .format(num_frames, width, height, lam, bg_diff, num_merged))
        targetfile = os.path.join(directory, 'simu_data_y_{}_{}_{}_lam_{}_diff_{}_merge_{}'
                        .format(num_frames, width, height, lam, bg_diff, num_merged))
        return datafile, targetfile









