import dataset.data.utils as utils
import dataset.data.constants as cons


def get_y_x_projection_shape(packet_shape):
    """
        Get the shape of a packet projection along the GTU axis derived from a
        packet of shape packet_shape.

        Parameters
        ----------
        :param packet_shape: shape of the original packet as a tuple of
                             (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :returns tuple of (frame height, frame width)
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return f_h, f_w


def get_gtu_x_projection_shape(packet_shape):
    """
        Get the shape of a packet projection along the Y axis derived from a
        packet of shape packet_shape.

        Parameters
        ----------
        :param packet_shape: shape of the original packet as a tuple of
                             (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :returns tuple of (number of frames, frame width)
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return n_f, f_w


def get_gtu_y_projection_shape(packet_shape):
    """
        Get the shape of a packet projection along the X axis derived from a
        packet of shape packet_shape.

        Parameters
        ----------
        :param packet_shape: shape of the original packet as a tuple of
                             (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :returns tuple of (number of frames, frame height)
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return n_f, f_h


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
        the eponymous argument and returns the result as a dict of str
        to tuple of int.

        The item_types parameter is a dict of str to boolean, where keys must
        be from the ALL_ITEM_TYPES constant and the value indicates if the
        shape should be calculated (True) or not (False) for a given item type.

        Parameters
        ----------
        :param packet_shape:  shape of the original packet as tuple of
                              (number of frames, frame height, frame width)
        :type packet_shape: tuple of int
        :param item_types: types of items for which their shape is to be
                           calculated.
        :type item_types: dict of (str,bool)
        :returns: dict of (str,tuple)
    """
    utils.check_item_types(item_types)
    return {k: (_item_shape_getters[k](packet_shape) if item_types[k]
                else None)
            for k in cons.ALL_ITEM_TYPES if item_types.get(k, False) is True}
