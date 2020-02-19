import numpy as np

import dataset.data.utils as utils
import dataset.data.constants as cons

# utility functions to convert raw packet data into other types of items


def create_subpacket(packet, start_idx=0, end_idx=None, dtype=None):
    """
        Convert a packet to a (sub)packet made up of frames from start_idx
        to end_idx (minus the latter), optionally casting to a new data type.

        Parameters
        ----------
        :param packet: original packet of recorded data
        :type packet: numpy.ndarray
        :param start_idx: index of first packet frame to include (default: 0)
        :type start_idx: int
        :param end_idx : index of first packet frame to exclude (default: None,
                         meaning include everything incl. the last frame)
        :type end_idx: int or None
        :param dtype: datatype of scalar values in new sub-packet
                      (default: None, meaning same as in original packet)
        :type dtype: numpy.dtype or str
        :returns numpy.ndarray:
    """
    if dtype is None:
        dtype = packet.dtype
    return packet[start_idx:end_idx].astype(dtype)


def create_y_x_projection_max(packet, start_idx=0, end_idx=None, dtype=None):
    """
        Convert packet to a projection by extracting the maximum of values
        along the GTU axis of the packet.

        Parameters optionally control the range of frames from which the
        projection is created and also the data type of scalar values in it.

        Parameters
        ----------
        :param packet: original packet of recorded data
        :type packet: numpy.ndarray
        :param start_idx: index of first packet frame to include (default: 0)
        :type start_idx: int
        :param end_idx : index of first packet frame to exclude (default: None,
                         meaning include everything incl. the last frame)
        :type end_idx: int or None
        :param dtype: datatype of scalar values in newly created projection
                      (default: None, meaning same as in original packet)
        :type dtype: numpy.dtype or str
        :returns numpy.ndarray:
    """
    return np.max(packet[start_idx:end_idx], axis=0).astype(dtype)


def create_gtu_x_projection_max(packet, start_idx=0, end_idx=None, dtype=None):
    """
        Convert packet to a projection by extracting the maximum of values
        along the Y axis of the packet.

        Parameters optionally control the range of frames from which the
        projection is created and also the data type of scalar values in it.

        Parameters
        ----------
        :param packet: original packet of recorded data
        :type packet: numpy.ndarray
        :param start_idx: index of first packet frame to include (default: 0)
        :type start_idx: int
        :param end_idx : index of first packet frame to exclude (default: None,
                         meaning include everything incl. the last frame)
        :type end_idx: int or None
        :param dtype: datatype of scalar values in newly created projection
                      (default: None, meaning same as in original packet)
        :type dtype: numpy.dtype or str
        :returns numpy.ndarray:
    """
    return np.max(packet[start_idx:end_idx], axis=1).astype(dtype)


def create_gtu_y_projection_max(packet, start_idx=0, end_idx=None, dtype=None):
    """
        Convert packet to a projection by extracting the maximum of values
        along the X axis of the packet.

        Parameters optionally control the range of frames from which the
        projection is created and also the data type of scalar values in it.

        Parameters
        ----------
        :param packet: original packet of recorded data
        :type packet: numpy.ndarray
        :param start_idx: index of first packet frame to include (default: 0)
        :type start_idx: int
        :param end_idx : index of first packet frame to exclude (default: None,
                         meaning include everything incl. the last frame)
        :type end_idx: int or None
        :param dtype: datatype of scalar values in newly created projection
                      (default: None, meaning same as in original packet)
        :type dtype: numpy.dtype or str
        :returns numpy.ndarray:
    """
    return np.max(packet[start_idx:end_idx], axis=2).astype(dtype)


_packet_converters = {
    'raw': create_subpacket,
    'yx': create_y_x_projection_max,
    'gtux': create_gtu_x_projection_max,
    'gtuy': create_gtu_y_projection_max
}


def convert_packet(packet, item_types, start_idx=0, end_idx=None,
                   dtype=None):
    """
        Convert packet to a set of data items as specified by the keys in the
        parameter item_types. This function serves as a wrapper which calls the
        appropriate converter function for all item types in the eponymous
        argument and returns the result as a dict of str to numpy.ndarray.

        The item_types parameter is a dict of str to boolean, where keys must
        be from the ALL_ITEM_TYPES constant and the value indicates if an item
        should be created (True) or not (False) for a given item type.

        Parameters
        ----------
        :param packet: original packet of recorded data
        :type packet: numpy.ndarray
        :param item_types: types of items for which derived item types are
                           to be created
        :type item_types: dict of (str,bool)
        :param start_idx: index of first packet frame to include (default: 0)
        :type start_idx: int
        :param end_idx : index of first packet frame to exclude (default: None,
                         meaning include everything incl. the last frame)
        :type end_idx: int or None
        :param dtype: datatype of scalar values in created derived items
                      (default: None, meaning same as in original packet)
        :type dtype: numpy.dtype or str
        :returns: dict of (str,numpy.ndarray)
    """
    utils.check_item_types(item_types)
    return {k: _packet_converters[k](packet, dtype=dtype, start_idx=start_idx,
                                     end_idx=end_idx)
            for k in cons.ALL_ITEM_TYPES if item_types.get(k, False) is True}
