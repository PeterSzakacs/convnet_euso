import argparse

def int_range(minval, maxval=None):
    def IntRange(value):
        try:
            val = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError('not an integer: '.format(value))
        if minval is not None and val < minval:
            raise argparse.ArgumentTypeError(
                'must be at least {} or more'.format(minval))
        if maxval is not None and val > maxval:
            raise argparse.ArgumentTypeError(
                'must be at least {} or more'.format(minval))
        return val
    return IntRange

def float_range(minval, maxval=None):
    def FloatRange(value):
        try:
            val = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError('not a float: '.format(value))
        if minval is not None and val < minval:
            raise argparse.ArgumentTypeError(
                'must be at least {} or more'.format(minval))
        if maxval is not None and val > maxval:
            raise argparse.ArgumentTypeError(
                'must be at least {} or more'.format(minval))
        return val
    return FloatRange