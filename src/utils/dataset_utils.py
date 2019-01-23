import operator
import functools

import numpy as np

import utils.metadata_utils as meta

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


def create_subpacket(packet, start_idx=0, end_idx=None):
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
    """
    return packet[start_idx:end_idx]


def create_y_x_projection(packet, start_idx=0, end_idx=None):
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
    """
    return np.max(packet[start_idx:end_idx], axis=0)


def create_gtu_x_projection(packet, start_idx=0, end_idx=None):
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
    """
    return np.max(packet[start_idx:end_idx], axis=1)


def create_gtu_y_projection(packet, start_idx=0, end_idx=None):
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
    """
    return np.max(packet[start_idx:end_idx], axis=2)


_packet_converters = {
    'raw': create_subpacket,
    'yx': create_y_x_projection,
    'gtux': create_gtu_x_projection,
    'gtuy': create_gtu_y_projection
}


def convert_packet(packet, item_types, start_idx=0, end_idx=None):
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
        item_types :     dict of str to bool
            the item types requested to be created from the original packet
    """
    check_item_types(item_types)
    return {k: (None if item_types[k] is False else
            _packet_converters[k](packet, start_idx, end_idx))
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


# classes


class numpy_dataset:
    """
        Class representing a dataset composed of data items of multiple types,
        all derived from packet data, and providing commonly used functionality
        for handling such datasets, e.g. getting and adding items, shuffling
        items etc.

        The types of items that can be stored here are raw packets as well as
        packet projectons.
    """

    def __init__(self, name, packet_shape, item_types={'raw': True,
                 'yx': False, 'gtux': False, 'gtuy': False}, dtype=np.uint8):
        check_item_types(item_types)
        self._item_types = item_types
        self._used_types = tuple(k for k in ALL_ITEM_TYPES
                                 if self._item_types[k] is True)
        self._item_shapes = get_data_item_shapes(packet_shape, item_types)
        self._packet_shape = tuple(packet_shape)
        self._data = create_data_holders(packet_shape, item_types=item_types,
                                         dtype=dtype)
        self._targets = []
        self._metadata = []
        self._metafields = set()
        self._num_data = 0
        self.resizable = True
        self.name = name

    def __str__(self):
        attrs_dict = {
            'name': self.name,
            'packet_shape': self.accepted_packet_shape,
            'item_types': self.item_types}
        return str(attrs_dict)

    # helper methods

    def _get_items_slice(self, items_slice_or_idx=None):
        return (slice(None) if items_slice_or_idx is None
                else items_slice_or_idx)

    # properties

    @property
    def name(self):
        """Name of the dataset."""
        return self._name

    @name.setter
    def name(self, value):
        """Name of the dataset."""
        self._name = value

    @property
    def resizable(self):
        """Boolean flag indicating whether adding items is allowed"""
        return self._resizable

    @resizable.setter
    def resizable(self, value):
        """Boolean flag indicating whether adding items is allowed"""
        self._resizable = value

    @property
    def num_data(self):
        """
            Current number of data in the dataset, cannot be more than capacity
            if it is not None.
        """
        return self._num_data

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

    @property
    def metadata_fields(self):
        """
            Fields of dataset data-level metadata.

            This is essentially a list or other sequence type which not only
            specifies the actual fields but also their order when the metadata
            are persisted as a TSV file.

            If metadata to be added with a new packet contains fields that are
            not in this property, they are added to the end of the sequence
            sorted by their names.
        """
        return self._metafields

    # add/get items

    def get_data_as_arraylike(self, data_slice_or_idx=None):
        s = self._get_items_slice(data_slice_or_idx)
        return tuple(self._data[k][s] for k in self._used_types)

    def get_data_as_dict(self, data_slice_or_idx=None):
        s = self._get_items_slice(data_slice_or_idx)
        return {k: self._data[k][s] for k in self._used_types}

    def get_targets(self, targets_slice_or_idx=None):
        s = self._get_items_slice(targets_slice_or_idx)
        return self._targets[s]

    def get_metadata(self, metadata_slice_or_idx=None):
        s = self._get_items_slice(metadata_slice_or_idx)
        return self._metadata[s]

    def add_data_item(self, packet, target, metadata={}):
        if not self._resizable:
            raise Exception('Cannot add items to dataset')
        if packet.shape != self.accepted_packet_shape:
            raise ValueError('Packet with incompatible shape passed.\n'
                             'Required:{}\nActual:'.format(self._packet_shape,
                                                           packet.shape))
        data_items = convert_packet(packet, item_types=self._item_types)
        for key in self._used_types:
            self._data[key].append(data_items[key])
        self._targets.append(target)
        self._metadata.append(metadata)
        self._metafields = self._metafields.union(metadata.keys())
        self._num_data += 1

    # dataset manipulation

    def shuffle_dataset(self, num_shuffles):
        """
            Shuffle dataset data, their targets and metadata in unison
            for a given number of times.

            Parameters
            ----------
            :param int num_shuffles:   number of times to shuffle the dataset
        """
        for idx in range(num_shuffles):
            rng_state = np.random.get_state()
            for key in self._used_types:
                np.random.shuffle(self._data[key])
                np.random.set_state(rng_state)
            np.random.shuffle(self._targets)
            np.random.set_state(rng_state)
            np.random.shuffle(self._metadata)

    def is_compatible_with(self, other_dataset):
        """
            Check if the other dataset is compatible with the current dataset.

            Essentially checks if these datasets have the same types of items
            and accept the same shape of packets.

            Parameters
            ----------
            :param other_dataset:   the dataset to check compatibility of
            :type other_dataset:    utils.dataset_utils.numpy_dataset
        """
        if other_dataset.item_types != self.item_types:
            return False
        if other_dataset.item_shapes != self.item_shapes:
            return False
        return True

    def merge_with(self, other_dataset, items_slice_or_idx=None):
        """
            Merge another dataset into the current dataset.

            Parameters
            ----------
            :param other_dataset:   the dataset to merge into the current one.
            :type other_dataset:    utils.dataset_utils.numpy_dataset
            :param items_slice_or_idx:  the slice of items from the dataset tp
                                        merge into the current dataset. Can be
                                        a simple numeric index or a slice.
            :type items_slice_or_idx:   int or slice
        """
        if not self._resizable:
            raise Exception('Cannot add items to this dataset')
        if not self.is_compatible_with(other_dataset):
            raise ValueError('Incompatible dataset to merge: {}',
                             other_dataset.name)
        s = self._get_items_slice(items_slice_or_idx)
        data = other_dataset.get_data_as_dict(s)
        for k in self._used_types:
            (self._data[k]).extend(data[k])
        targets = other_dataset.get_targets(s)
        self._targets.extend(targets)
        metadata = other_dataset.get_metadata(s)
        self._metadata.extend(metadata)
        if s == slice(None):
            metafields = other_dataset._metafields
        else:
            metafields = meta.extract_metafields(metadata)
        self._metafields = self._metafields.union(metafields)
        self._num_data += len(metadata)

    def add_metafield(self, name, default_value=None):
        """
            Add a new metafield to the dataset.

            Parameters
            ----------
            :param name:   name of the field
            :type name:    str
            :param default_value:  default value for the new metafield
            :type default_value:   any
        """
        self._metafields = self._metafields.union([name])
        for metadata in self._metadata:
            metadata[name] = default_value
