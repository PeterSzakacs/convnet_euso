import collections

import numpy as np

# functions to check if a passed value represents an interval with an upper and lower bound
# legal values include:
# - (tuple or list) of int (length 2)
# - numpy.ndarray with shape (2)
# - range(object) -> the upper and lower bounds correspond to start and stop properties
# all these are coonverted to tuples

def check_and_convert_value_to_tuple(value, property_name):
    isndarr, isrange = isinstance(value, np.ndarray), isinstance(value, range)
    isvalidtype = isndarr or isrange or isinstance(value, collections.Sequence)
    if isinstance(value, str) or not isvalidtype:
        raise TypeError(('Incorrect type for property {}, expected Sequence type (except basestr and subclasses)'
                        ' or numpy array with shape (2), got: {}').format(property_name, type(value)))
    if isrange:
        return (value.start, value.stop)
    elif isndarr:
        if len(value.shape) > 1:
            raise ValueError(('Incorrect shape of numpy array passed as property {}, expected: (2), got: {}'.format(
                                property_name, value.shape)))
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(('Incorrect element type for numpy array passed as property {}, expected: {}, got: {}'.format(
                                property_name, 'integer integer', value.shape)))
    seq_len = len(value)
    if seq_len != 2:
        raise ValueError(('Incorrect length of sequence for property {}, expected 2 values (min, max), got: {} values'.format(
                                property_name, seq_len)))
    return (value[0], value[1])

def check_interval_tuple(interval_tuple, property_name, lower_limit=None, upper_limit=None):
    lower_limit = interval_tuple[0] if lower_limit == None else lower_limit
    upper_limit = interval_tuple[1] if upper_limit == None else upper_limit
    if interval_tuple[0] < lower_limit:
        raise ValueError('Lower bound for property {} must be greater than or eual to {}'.format(property_name, lower_limit))
    if interval_tuple[1] < interval_tuple[0]:
        raise ValueError('Upper bound of property {} must be greater than or equal to its lower bound'.format(property_name))
    if interval_tuple[1] > upper_limit:
        raise ValueError('Upper bound of property {} must not be greater than {}'.format(property_name, upper_limit))