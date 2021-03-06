import collections

import numpy as np

# functions to check if a passed value represents an interval with an upper and
# lower bound. Legal types of value include:
# - (tuple or list) of int (length 2)
# - numpy.ndarray with shape (2)
# - range(object) -> the upper and lower bounds correspond to start and stop p
#                    roperties
# all these are coonverted to tuples


def check_and_convert_value_to_tuple(value, property_name):
    isndarr, isrange = isinstance(value, np.ndarray), isinstance(value, range)
    isvalidtype = isndarr or isrange or isinstance(value, collections.Sequence)
    if isinstance(value, str) or not isvalidtype:
        raise TypeError(('Incorrect type for property {}, expected Sequence'
                         ' type (except basestr and subclasses) or numpy array'
                         ' with shape (2), got: {}').format(property_name,
                                                            type(value)))
    if isrange:
        return (value.start, value.stop)
    elif isndarr:
        if len(value.shape) > 1:
            raise ValueError(('Incorrect shape of numpy array passed as'
                              ' property {}, expected: (2), got: {}').format(
                              property_name, value.shape))
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(('Incorrect element type for numpy array passed as'
                             ' property {}, expected: {}, got: {}').format(
                             property_name, 'integer', type(value)))
    seq_len = len(value)
    if seq_len != 2:
        raise ValueError(('Incorrect length of sequence for property {},'
                         ' expected 2 values (min, max), got: {} values')
                         .format(property_name, seq_len))
    return (value[0], value[1])


def check_interval_tuple(interval_tuple, property_name, lower_limit=None,
                         upper_limit=None):
    lower_limit = lower_limit or interval_tuple[0]
    upper_limit = upper_limit or interval_tuple[1]
    if interval_tuple[0] < lower_limit:
        raise ValueError('Lower bound for property {} must be greater than'
                         ' or eual to {}'.format(property_name, lower_limit))
    if interval_tuple[1] < interval_tuple[0]:
        raise ValueError('Upper bound of property {} must be greater than'
                         ' or equal to its lower bound'.format(property_name))
    if interval_tuple[1] > upper_limit:
        raise ValueError('Upper bound of property {} must not be greater than'
                         ' {}'.format(property_name, upper_limit))

# function to create a conversion/cast function from string to type 'typename',
# optionally also applying rules for floating-point precision and handling null
# values (defined as empty string '').
#
# Currently only supports string, int and float target types.


SUPOORTED_CAST_TYPES=('str', 'int', 'float', )


def get_cast_func(typename, fp_precision=None, nullable=False,
                  default_value=''):
    if typename == 'str':
        fn = str
    elif typename == 'int':
        fn = int
    elif typename == 'float':
        if fp_precision is None:
            fn = float
        else:
            fn = lambda val: round(float(val), fp_precision)
    else:
        raise ValueError('Unsupported type: {}'.format(typename))
    if nullable:
        return lambda val: default_value if val == '' else fn(val)
    else:
        return fn

# equality implementation for classes primarily intended to hold simple values
# (templates)


class CommonEqualityMixin(object):

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
