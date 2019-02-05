import operator
import functools

import numpy as np

# module level constants

ALL_ITEM_TYPES = ('raw', 'yx', 'gtux', 'gtuy')

# other functions


def check_item_types(item_types):
    # all keys are set to false
    if not functools.reduce(operator.or_, item_types.values(), False):
        raise ValueError(('At least one item type (possible types: {})'
                         ' must be used in the dataset').format(
                         ALL_ITEM_TYPES))
    illegal_keys = item_types.keys() - set(ALL_ITEM_TYPES)
    if len(illegal_keys) > 0:
        raise Exception(('Unknown keys found: {}'.format(illegal_keys)))


# holder creation


def create_packet_holder(packet_shape, num_items=None, dtype=np.uint8):
    """
        Create a data structure for holding raw packets with shape packet_shape
        and able to hold either an unlimited or at most num_items number of
        items.

        Parameters
        ----------
        packet_shape :  tuple of 3 ints
            shape of the original packet
        num_items :     int or None
            expected number of items which will be stored in the holder
            or None if not known in advance
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_h, f_w), dtype=dtype)


def create_y_x_projection_holder(packet_shape, num_items=None, dtype=np.uint8):
    """
        Create a data structure for holding packet projections along the GTU
        axis created from packets with shape packet_shape and able to hold
        either an unlimited or at most num_items number of items.

        Parameters
        ----------
        packet_shape :  tuple of 3 ints
            shape of the original packet
        num_items :     int or None
            expected number of items which will be stored in the holder
            or None if not known in advance
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, f_h, f_w), dtype=dtype)


def create_gtu_x_projection_holder(packet_shape, num_items=None,
                                   dtype=np.uint8):
    """
        Create a data structure for holding packet projections along the Y
        axis created from packets with shape packet_shape and able to hold
        either an unlimited or at most num_items number of items.

        Parameters
        ----------
        packet_shape :  tuple of 3 ints
            shape of the original packet
        num_items :     int or None
            expected number of items which will be stored in the holder
            or None if not known in advance
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_w), dtype=dtype)


def create_gtu_y_projection_holder(packet_shape, num_items=None,
                                   dtype=np.uint8):
    """
        Create a data structure for holding packet projections along the X
        axis created from packets with shape packet_shape and able to hold
        either an unlimited or at most num_items number of items.

        Parameters
        ----------
        packet_shape :  tuple of 3 ints
            shape of the original packet
        num_items :     int or None
            expected number of items which will be stored in the holder
            or None if not known in advance
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_h), dtype=dtype)


_holder_creators = {
    'raw': create_packet_holder,
    'yx': create_y_x_projection_holder,
    'gtux': create_gtu_x_projection_holder,
    'gtuy': create_gtu_y_projection_holder
}


def create_data_holders(packet_shape, item_types, num_items=None,
                        dtype=np.uint8):
    """
        Create a collection of data structures for holding data items specified
        in item_types (with these items being created from packets with shape
        packet_shape) each of these structures able to hold either an unlimited
        or at most num_items number of items. This function serves as a wrapper
        to call the respective holder creators for a given item type and return
        these holders in the form of a dict of str to array-like. Where an item
        type is set to False, the value for the same key in the returned dict
        is None.

        Parameters
        ----------
        packet_shape :  tuple of 3 ints
            shape of the original packet
        num_items :     int or None
            expected number of items which will be stored in the holder
            or None if not known in advance
    """
    check_item_types(item_types)
    return {k: (None if item_types[k] is False else
            _holder_creators[k](packet_shape, num_items, dtype))
            for k in ALL_ITEM_TYPES}

# data item creation


def create_subpacket(packet, start_idx=0, end_idx=None, dtype=np.uint8):
    """
        Convert packet to a (sub)packet made up of frames from start_idx
        to end_idx (minus the latter).

        Parameters
        ----------
        packet :        3-dimensional numpy.ndarray
            packet from which to create a (sub)packet
        start_idx :     int
            index of first frame in the packet to inculde
        end_idx :       int or None
            index of first frame in the packet to exclude
        dtype :         str or np.number
            data type of created subpacket
    """
    return packet[start_idx:end_idx].astype(dtype)


def create_y_x_projection(packet, start_idx=0, end_idx=None, dtype=np.uint8):
    """
        Convert packet to a projection by extracting the maximum of values
        along the GTU axis of the packet made up of frames from start_idx
        to end_idx (minus the latter).

        Parameters
        ----------
        packet :        3-dimensional numpy.ndarray
            packet from which to create the projection
        start_idx :     int
            index of first packet frame to use in creating the projection
        end_idx :       int or None
            index of first packet frame to not use in creating the projection
        dtype :         str or np.number
            data type of created yx projection
    """
    return np.max(packet[start_idx:end_idx], axis=0).astype(dtype)


def create_gtu_x_projection(packet, start_idx=0, end_idx=None, dtype=np.uint8):
    """
        Convert packet to a projection by extracting the maximum of values
        along the Y axis of the packet made up of frames from start_idx to
        end_idx (minus the latter).

        Parameters
        ----------
        packet :        3-dimensional numpy.ndarray
            packet from which to create the projection
        start_idx :     int
            index of first packet frame to use in creating the projection
        end_idx :       int or None
            index of first packet frame to not use in creating the projection
        dtype :         str or np.number
            data type of created gtux projection
    """
    return np.max(packet[start_idx:end_idx], axis=1).astype(dtype)


def create_gtu_y_projection(packet, start_idx=0, end_idx=None, dtype=np.uint8):
    """
        Convert packet to a projection by extracting the maximum of values
        along the X axis of the packet made up of frames from start_idx to
        end_idx (minus the latter).

        Parameters
        ----------
        packet :        3-dimensional numpy.ndarray
            packet from which to create the projection
        start_idx :     int
            index of first packet frame to use in creating the projection
        end_idx :       int or None
            index of first packet frame to not use in creating the projection
        dtype :         str or np.number
            data type of created gtuy projection
    """
    return np.max(packet[start_idx:end_idx], axis=2).astype(dtype)


_packet_converters = {
    'raw': create_subpacket,
    'yx': create_y_x_projection,
    'gtux': create_gtu_x_projection,
    'gtuy': create_gtu_y_projection
}


def convert_packet(packet, item_types, start_idx=0, end_idx=None,
                   dtype=np.uint8):
    """
        Convert packet to a set of data items as specified by the keys in the
        parameter item_types. This function serves as a wrapper which calls the
        appropriate packet conversion fuction for each item type specified in
        item_types and returns the result as a dict of str to ndarray. Where
        the item type is set to False, the value for the same key in the
        returned dict is None.

        Parameters
        ----------
        packet :        3-dimensional numpy.ndarray
            packet from which to create a projection along the y axis
        start_idx :     int
            index of first packet frame to use in creating the data items
        end_idx :       int or None
            index of first packet frame to not use in creating the data itmes
        item_types :    dict of str to bool
            the item types requested to be created from the original packet
    """
    check_item_types(item_types)
    return {k: (None if item_types[k] is False else _packet_converters[k](
                    packet, dtype=dtype, start_idx=start_idx, end_idx=end_idx))
            for k in ALL_ITEM_TYPES}

# get data item shape


def get_y_x_projection_shape(packet_shape):
    """
        Get the shape of a packet projection along the GTU axis derived from a
        packet of shape packet_shape.

        Parameters
        ----------
        packet_shape :      tuple of 3 ints
            shape of the packet from which the projection is created
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return (f_h, f_w)


def get_gtu_x_projection_shape(packet_shape):
    """
        Get the shape of a packet projection along the Y axis derived from a
        packet of shape packet_shape.

        Parameters
        ----------
        packet_shape :      tuple of 3 ints
            shape of the packet from which the projection is created
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return (n_f, f_w)


def get_gtu_y_projection_shape(packet_shape):
    """
        Get the shape of a packet projection along the X axis derived from a
        packet of shape packet_shape.

        Parameters
        ----------
        packet_shape :      tuple of 3 ints
            shape of the packet from which the projection is created
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return (n_f, f_h)


_item_shape_getters = {
    'raw': lambda packet_shape: packet_shape,
    'yx': get_y_x_projection_shape,
    'gtux': get_gtu_x_projection_shape,
    'gtuy': get_gtu_y_projection_shape
}


def get_data_item_shapes(packet_shape, item_types):
    """
        Get the shapes of items of the given types derived from a packet
        of shape packet_shape. This function serves as a wrapper which
        calls the appropriate item shape getters for all item types in
        the eponymous arguemnt and returns the result as a dict of str
        to tuple of int. Where the item type is set to False, the value
        for the same key in the returned dict is None.

        Parameters
        ----------
        packet_shape :      tuple of 3 ints
            shape of the packet from which the projection is created
        item_types :     dict of str to bool
            the item types for which their shapes are requested
    """
    check_item_types(item_types)
    return {k: (None if item_types[k] is False else
            _item_shape_getters[k](packet_shape))
            for k in ALL_ITEM_TYPES}
