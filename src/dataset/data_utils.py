import collections
import functools
import operator
import typing as t

import numpy as np

import dataset.constants as cons


# other functions


def check_item_types(item_types):
    # all keys are set to false
    if not functools.reduce(operator.or_, item_types.values(), False):
        raise ValueError(('At least one item type (possible types: {})'
                         ' must be used in the dataset').format(
                         cons.ALL_ITEM_TYPES))
    illegal_keys = item_types.keys() - set(cons.ALL_ITEM_TYPES)
    if len(illegal_keys) > 0:
        raise Exception(('Unknown keys found: {}'.format(illegal_keys)))


# holder creation


def create_packet_holder(
        packet_shape: t.Sequence[int],
        num_items: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> t.Union[list, np.ndarray]:
    """
        Create a data structure for holding raw packets with shape packet_shape
        and able to hold either an unlimited or at most num_items number of
        items.

        :param packet_shape: shape of the original packet
        :param num_items: (optional) expected number of items which will be
                          stored in the holder or None if not known in advance
        :param dtype: (optional) dtype of the items in the holder - only
                      important if num_items is not None
        :return: empty holder for packets
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_h, f_w), dtype=dtype)


def create_y_x_projection_holder(
        packet_shape: t.Sequence[int],
        num_items: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> t.Union[list, np.ndarray]:
    """
        Create a data structure for holding packet projections along the GTU
        axis created from packets with shape packet_shape and able to hold
        either an unlimited or at most num_items number of items.

        :param packet_shape: shape of the original packet
        :param num_items: (optional) expected number of items which will be
                          stored in the holder or None if not known in advance
        :param dtype: (optional) dtype of the items in the holder - only
                      important if num_items is not None
        :return: empty holder for yx projections
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, f_h, f_w), dtype=dtype)


def create_gtu_x_projection_holder(
        packet_shape: t.Sequence[int],
        num_items: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> t.Union[list, np.ndarray]:
    """
        Create a data structure for holding packet projections along the Y
        axis created from packets with shape packet_shape and able to hold
        either an unlimited or at most num_items number of items.

        :param packet_shape: shape of the original packet
        :param num_items: (optional) expected number of items which will be
                          stored in the holder or None if not known in advance
        :param dtype: (optional) dtype of the items in the holder - only
                      important if num_items is not None
        :return: empty holder for gtux projections
    """
    n_f, f_h, f_w = packet_shape[0:3]
    if num_items is None:
        return []
    else:
        return np.empty((num_items, n_f, f_w), dtype=dtype)


def create_gtu_y_projection_holder(
        packet_shape: t.Sequence[int],
        num_items: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> t.Union[list, np.ndarray]:
    """
        Create a data structure for holding packet projections along the X
        axis created from packets with shape packet_shape and able to hold
        either an unlimited or at most num_items number of items.

        :param packet_shape: shape of the original packet
        :param num_items: (optional) expected number of items which will be
                          stored in the holder or None if not known in advance
        :param dtype: (optional) dtype of the items in the holder - only
                      important if num_items is not None
        :return: empty holder for gtuy projections
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


def create_data_holders(
        packet_shape: t.Sequence[int],
        item_types: t.Union[t.Mapping[str, bool], t.Iterable[str]],
        num_items: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> t.Union[t.Mapping[str, list], t.Mapping[str, np.ndarray]]:
    """
        Create a collection of data structures for holding data items specified
        in item_types (with these items being created from packets with shape
        packet_shape) each of these structures able to hold either an unlimited
        or at most num_items number of items. This function serves as a wrapper
        to call the respective holder creators for a given item type and return
        these holders in the form of a dict of str to array-like. Where an item
        type is set to False, the value for the same key in the returned dict
        is None.

        :param packet_shape: shape of the original packet
        :param item_types: the item types for which holders are to be created
        :param num_items: (optional) expected number of items which will be
                          stored in the holder or None if not known in advance
        :param dtype: (optional) dtype of the items in the holder(s) - only
                      important if num_items is not None
        :return: mapping of empty holders indexed by the held item type
                 name/key
    """
    if isinstance(item_types, t.Mapping):
        check_item_types(item_types)
        return {k: (None if item_types[k] is False else
                _holder_creators[k](packet_shape, num_items=num_items,
                                    dtype=dtype))
                for k in cons.ALL_ITEM_TYPES}
    else:
        _types = dict.fromkeys(item_types, True)
        check_item_types(_types)
        return {k: _holder_creators[k](packet_shape, num_items=num_items,
                                       dtype=dtype)
                for k in _types.keys()}


# data item creation


def create_subpacket(
        packet: np.ndarray,
        start_idx: int = 0,
        end_idx: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> np.ndarray:
    """
        Convert packet to a (sub)packet made up of frames from start_idx
        to end_idx (minus the latter).

        :param packet: packet from which to create the (sub)packet
        :param start_idx: (optional) index of first packet frame to use for
                          creating the (sub)packet
        :param end_idx: (optional) index of first packet frame to NOT use for
                        creating the (sub)packet (same semantics as 'stop'
                        param for e.g. range() objects)
        :param dtype: (optional) data type to cast the created (sub)packet to
        :return: (sub)packet derived from the passed packet
    """
    return packet[start_idx:end_idx].astype(dtype)


def create_y_x_projection(
        packet: np.ndarray,
        start_idx: int = 0,
        end_idx: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> np.ndarray:
    """
        Convert packet to a projection by extracting the maximum of values
        along the GTU axis of the packet made up of frames from start_idx
        to end_idx (minus the latter).

        :param packet: packet from which to create the projection
        :param start_idx: (optional) index of first packet frame to use for
                          creating the projection
        :param end_idx: (optional) index of first packet frame to NOT use for
                        creating the projection (same semantics as 'stop' param
                        for e.g. range() objects)
        :param dtype: (optional) data type to cast the created projection to
        :return: projection derived from the passed packet
    """
    return np.max(packet[start_idx:end_idx], axis=0).astype(dtype)


def create_gtu_x_projection(
        packet: np.ndarray,
        start_idx: int = 0,
        end_idx: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> np.ndarray:
    """
        Convert packet to a projection by extracting the maximum of values
        along the Y axis of the packet made up of frames from start_idx to
        end_idx (minus the latter).

        :param packet: packet from which to create the projection
        :param start_idx: (optional) index of first packet frame to use for
                          creating the projection
        :param end_idx: (optional) index of first packet frame to NOT use for
                        creating the projection (same semantics as 'stop' param
                        for e.g. range() objects)
        :param dtype: (optional) data type to cast the created projection to
        :return: projection derived from the passed packet
    """
    return np.max(packet[start_idx:end_idx], axis=1).astype(dtype)


def create_gtu_y_projection(
        packet: np.ndarray,
        start_idx: int = 0,
        end_idx: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> np.ndarray:
    """
        Convert packet to a projection by extracting the maximum of values
        along the X axis of the packet made up of frames from start_idx to
        end_idx (minus the latter).

        :param packet: packet from which to create the projection
        :param start_idx: (optional) index of first packet frame to use for
                          creating the projection
        :param end_idx: (optional) index of first packet frame to NOT use for
                        creating the projection (same semantics as 'stop' param
                        for e.g. range() objects)
        :param dtype: (optional) data type to cast the created projection to
        :return: projection derived from the passed packet
    """
    return np.max(packet[start_idx:end_idx], axis=2).astype(dtype)


_packet_converters = {
    'raw': create_subpacket,
    'yx': create_y_x_projection,
    'gtux': create_gtu_x_projection,
    'gtuy': create_gtu_y_projection
}


def convert_packet(
        packet: np.ndarray,
        item_types: t.Union[t.Mapping[str, bool], t.Iterable[str]],
        start_idx: int = 0,
        end_idx: int = None,
        dtype: t.Union[str, np.dtype] = np.uint8
) -> t.Mapping[str, np.ndarray]:
    """
        Convert packet to a set of data items as specified by the keys in the
        parameter item_types. This function serves as a wrapper which calls the
        appropriate packet conversion fuction for each item type specified in
        item_types and returns the result as a dict of str to ndarray. Where
        the item type is set to False, the value for the same key in the
        returned dict is None.

        :param packet: packet from which to create derived item types
        :param item_types: the item types requested to be created from the
                           original packet
        :param start_idx: (optional) index of first packet frame to use for
                          creating the derived items
        :param end_idx: (optional) index of first packet frame to NOT use for
                        creating the itmes (same semantics as 'stop' param
                        for e.g. range() objects)
        :param dtype: (optional) data type to cast the created items to
        :return: mapping of items of requested types derived from the packet
                 indexed by item type name/key
    """
    if isinstance(item_types, t.Mapping):
        check_item_types(item_types)
        return {k: (None if item_types[k] is False
                    else _packet_converters[k](packet, dtype=dtype,
                                               start_idx=start_idx,
                                               end_idx=end_idx))
                for k in cons.ALL_ITEM_TYPES}
    else:
        _types = dict.fromkeys(item_types, True)
        check_item_types(_types)
        return {k: _packet_converters[k](packet, dtype=dtype,
                                         start_idx=start_idx,
                                         end_idx=end_idx)
                for k in _types.keys()}


# get data item shape


def get_y_x_projection_shape(
        packet_shape: t.Sequence[int]
) -> t.Sequence[int]:
    """
        Get the shape of a packet projection along the GTU axis derived from a
        packet of shape packet_shape.

        :param packet_shape: shape of the packet from which the projection is
                             created
        :return: shape of the projection
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return (f_h, f_w)


def get_gtu_x_projection_shape(
        packet_shape: t.Sequence[int]
) -> t.Sequence[int]:
    """
        Get the shape of a packet projection along the Y axis derived from a
        packet of shape packet_shape.

        :param packet_shape: shape of the packet from which the projection is
                             created
        :return: shape of the projection
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return (n_f, f_w)


def get_gtu_y_projection_shape(
        packet_shape: t.Sequence[int]
) -> t.Sequence[int]:
    """
        Get the shape of a packet projection along the X axis derived from a
        packet of shape packet_shape.

        :param packet_shape: shape of the packet from which the projection is
                             created
        :return: shape of the projection
    """
    n_f, f_h, f_w = packet_shape[0:3]
    return (n_f, f_h)


_item_shape_getters = {
    'raw': lambda packet_shape: packet_shape,
    'yx': get_y_x_projection_shape,
    'gtux': get_gtu_x_projection_shape,
    'gtuy': get_gtu_y_projection_shape
}


def get_data_item_shapes(
        packet_shape: t.Sequence[int],
        item_types: t.Union[t.Mapping[str, bool], t.Iterable[str]],
) -> t.Mapping[str, t.Sequence[int]]:
    """
        Get the shapes of items of the given types derived from a packet
        of shape packet_shape. This function serves as a wrapper which
        calls the appropriate item shape getters for all item types in
        the eponymous arguemnt and returns the result as a dict of str
        to tuple of int. Where the item type is set to False, the value
        for the same key in the returned dict is None.

        :param packet_shape: shape of the packet from which the specified item
                             types are derived
        :param item_types: item types for which their shapes are requested
        :return: mapping of shapes for items derived from a packet with the
                 given shape indexed by item type name/key
    """
    if isinstance(item_types, t.Mapping):
        check_item_types(item_types)
        return {k: (None if item_types[k] is False else
                _item_shape_getters[k](packet_shape))
                for k in cons.ALL_ITEM_TYPES}
    else:
        _types = dict.fromkeys(item_types, True)
        check_item_types(_types)
        return {k: _item_shape_getters[k](packet_shape) for k in _types.keys()}


# classes

class DataHolder():

    def __init__(self, packet_shape, dtype=np.uint8, item_types={'raw': True,
                 'yx': False, 'gtux': False, 'gtuy': False}):
        check_item_types(item_types)
        self._num_items = 0
        self._item_types = item_types
        self._used_types = tuple(k for k in cons.ALL_ITEM_TYPES
                                 if item_types[k] is True)
        self._item_shapes = get_data_item_shapes(packet_shape, item_types)
        self._packet_shape = tuple(packet_shape)
        self._data = create_data_holders(packet_shape, dtype=dtype,
                                         item_types=item_types)
        self.dtype = dtype

    def __len__(self):
        return self._num_items

    def _get_indexes_sequence(self, indexing_obj):
        if indexing_obj is None:
            return range(self._num_items)
        elif isinstance(indexing_obj, slice):
            return range(self._num_items)[indexing_obj]
        elif isinstance(indexing_obj, collections.Sequence):
            # range, list, tuple, etc
            return indexing_obj

    @property
    def dtype(self):
        """Datatype of packet data and derived items."""
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """Datatype of packet data and derived items."""
        dtype = np.dtype(value)
        if not np.issubdtype(dtype, np.number):
            raise Exception('Illegal data type: {}'.format(value))
        for item_type in self._used_types:
            data = self._data[item_type]
            data = [datum.astype(dtype) for datum in data]
            self._data[item_type] = data
        self._dtype = dtype.name

    @property
    def item_types(self):
        """
            The type of items in this dataset, as a dict of str to bool, where
            the all the keys are from the 'ALL_ITEM_TYPES' module constant and
            the values represent wheher a collection of items of this type is
            present in this dataset.
        """
        return self._item_types

    @property
    def accepted_packet_shape(self):
        """
            The shape of accepted raw packets to be converted to dataset items.
        """
        return self._packet_shape

    @property
    def item_shapes(self):
        """
            The shape of individual item types in this dataset, in form of a
            dict of str to tuple of int, where the keys are from the module
            constant 'ALL_ITEM_TYPES' and the values represent the shape of
            items as they would be presented from the items themselves by
            getting the numpy.ndarray 'shape' property.
        """
        return self._item_shapes

    def append(self, items_dict):
        used_types = self._used_types
        missing = set(itype for itype in used_types
                      if items_dict.get(itype, None) is None)
        if missing:
            raise Exception('Missing item types detected: {}'.format(missing))
        for itype in self._used_types:
            self._data[itype].append(items_dict[itype].astype(self.dtype))
        self._num_items += 1

    def extend(self, items_iter_dict):
        used_types = self._used_types
        missing = set(itype for itype in used_types
                      if items_iter_dict.get(itype, None) is None)
        if missing:
            raise Exception('Missing item types detected: {}'.format(missing))
        for itype in used_types:
            self._data[itype].extend(
                item.astype(self.dtype) for item in items_iter_dict[itype])
        self._num_items = len(self._data[used_types[0]])

    def append_packet(self, packet):
        s = packet.shape
        if s != self.accepted_packet_shape:
            raise ValueError('Wrong packet shape passed. Expected. {}, '
                             'actual: {}'.format(self._packet_shape, s))
        self.append(convert_packet(packet, self.item_types, dtype=self.dtype))

    def extend_packets(self, packets_iter):
        for packet in packets_iter:
            self.append_packet(packet)

    def get_data_as_arraylike(self, data_slice_or_idx=None):
        idxs = self._get_indexes_sequence(data_slice_or_idx)
        data = self._data
        return tuple([data[k][idx] for idx in idxs] for k in self._used_types)

    def get_data_as_dict(self, data_slice_or_idx=None):
        idxs = self._get_indexes_sequence(data_slice_or_idx)
        data = self._data
        return {k: [data[k][idx] for idx in idxs] for k in self._used_types}

    def shuffle(self, shuffler, shuffler_state_resetter):
        for item_type in self._used_types:
            items = self._data[item_type]
            shuffler(items)
            shuffler_state_resetter()
