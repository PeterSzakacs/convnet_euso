import numpy as np

import dataset.data.constants as cons
import dataset.data.utils as utils


def create_packet_container(packet_shape, num_items=None, dtype=np.uint8):
    """
        Create an array-like data structure to contain raw packets of data
        with shape packet_shape.

        If num_items is specified, returns a numpy array able to hold at most 
        num_items number of items of shape packet shape, else returns just an 
        empty list.
        
        This function is a wrapper utility to help create data item containers
        for a new dataset depending on whether it should be easily resizable
        and what is the scalar type of the contained data items.

        Parameters
        ----------
        :param packet_shape:  shape of the original packet as tuple of
                              (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :param num_items: expected number of items to store or None if not
                          known in advance
        :type num_items: int or None
        :param dtype: scalar data type of stored items (default: uint8)
        :type dtype: numpy.dtype or str
        :return list or numpy.ndarray:
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_h, f_w), dtype=dtype)


def create_y_x_projection_container(packet_shape, num_items=None,
                                    dtype=np.uint8):
    """
        Create an array-like data structure to contain projections of packet
        data along the GTU axis (created from packets with shape packet_shape).

        If num_items is specified, returns a numpy array able to hold at most
        num_items number of items of shape derived from packet_shape, else
        returns just an empty list.

        This function is a wrapper utility to help create data item containers
        for a new dataset depending on whether it should be easily resizable
        and what is the scalar type of the contained data items.

        Parameters
        ----------
        :param packet_shape:  shape of the original packet as tuple of
                              (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :param num_items: expected number of items to store or None if not
                          known in advance
        :type num_items: int or None
        :param dtype: scalar data type of stored items (default: uint8)
        :type dtype: numpy.dtype or str
        :return list or numpy.ndarray:
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, f_h, f_w), dtype=dtype)


def create_gtu_x_projection_container(packet_shape, num_items=None,
                                      dtype=np.uint8):
    """
        Create an array-like data structure to contain projections of packet
        data along the Y axis (created from packets with shape packet_shape).

        If num_items is specified, returns a numpy array able to hold at most
        num_items number of items (with shape derived from packet_shape), else
        returns just an empty list.

        This function is a wrapper utility to help create data item containers
        for a new dataset depending on whether it should be easily resizable
        and what is the scalar type of the contained data items.

        Parameters
        ----------
        :param packet_shape:  shape of the original packet as tuple of
                              (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :param num_items: expected number of items to store or None if not
                          known in advance
        :type num_items: int or None
        :param dtype: scalar data type of stored items (default: uint8)
        :type dtype: numpy.dtype or str
        :return list or numpy.ndarray:
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_w), dtype=dtype)


def create_gtu_y_projection_container(packet_shape, num_items=None,
                                      dtype=np.uint8):
    """
        Create an array-like data structure to contain projections of packet
        data along the X axis (created from packets with shape packet_shape).

        If num_items is specified, returns a numpy array able to hold at most
        num_items number of items (with shape derived from packet_shape), else
        returns just an empty list.

        This function is a wrapper utility to help create data item containers
        for a new dataset depending on whether it should be easily resizable
        and what is the scalar type of the contained data items.

        Parameters
        ----------
        :param packet_shape:  shape of the original packet as tuple of
                              (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :param num_items: expected number of items to store or None if not
                          known in advance
        :type num_items: int or None
        :param dtype: scalar data type of stored items (default: uint8)
        :type dtype: numpy.dtype or str
        :return list or numpy.ndarray:
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_h), dtype=dtype)


_holder_creators = {
    'raw': create_packet_container,
    'yx': create_y_x_projection_container,
    'gtux': create_gtu_x_projection_container,
    'gtuy': create_gtu_y_projection_container
}


def create_data_containers(packet_shape, item_types, num_items=None,
                           dtype=np.uint8):
    """
        Create a collection of data structures to contain data items specified
        in item_types.

        This function serves as a wrapper to call the container creators for
        all requested item types and return them in the form of a dict of str
        to array-like.

        The item_types parameter is a dict of str to boolean, where keys must
        be from the ALL_ITEM_TYPES constant and the value indicates if the
        holder should be created (True) or not (False) for a given item type.

        Parameters
        ----------
        :param packet_shape:  shape of the original packet as tuple of
                              (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :param item_types: types of items for which containers are to be
                           created
        :type item_types: dict of (str,bool)
        :param num_items: expected number of items to store or None if not
                          known in advance
        :type num_items: int or None
        :param dtype: scalar data type of stored items (default: uint8)
        :type dtype: numpy.dtype or str
        :return dict of (str,list or numpy.ndarray):
    """
    utils.check_item_types(item_types)
    return {k: _holder_creators[k](packet_shape, num_items, dtype)
            for k in cons.ALL_ITEM_TYPES if item_types.get(k, False) is True}
