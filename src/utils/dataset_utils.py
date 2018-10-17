import os
import csv
import configparser
import operator
import functools

import numpy as np

# module level constants

ALL_ITEM_TYPES = ('raw', 'yx', 'gtux', 'gtuy')

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


def create_data_holders(packet_shape, num_items=None, dtype=np.uint8,
                        item_types={'raw': True, 'yx': False, 'gtux': False,
                        'gtuy': False}):
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


def convert_packet(packet, start_idx=0, end_idx=None, item_types={'raw': True,
                   'yx': False, 'gtux': False, 'gtuy': False}):
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


def get_data_item_shapes(packet_shape, item_types={'raw': True, 'yx': False,
                         'gtux': False, 'gtuy': False}):
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
    return {k: (None if item_types[k] is False else
            _item_shape_getters[k](packet_shape))
            for k in ALL_ITEM_TYPES}

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


def get_train_and_test_sets(dataset, test_num_items=None, test_fraction=0.1):
    num_data = dataset.num_data
    if test_fraction > 1:
        raise ValueError(('Requested an evaluation set from original dataset,'
                          ' that is {}% the size of the original').format(
                          test_fraction*100))
    if test_num_items is not None and test_num_items > num_data:
        raise ValueError(('The number of items to select from the dataset ({})'
                          '  is larger than the dataset ({})').format(
                          test_num_items, dataset.size))
    test_num = test_num_items or round(test_fraction * num_data)
    train_data = dataset.get_data_as_dict(slice(test_num, num_data))
    train_targets = dataset.get_targets(slice(test_num, num_data))

    test_data = dataset.get_data_as_dict(slice(test_num))
    test_targets = dataset.get_targets(slice(test_num))
    return train_data, train_targets, test_data, test_targets

# classes


class numpy_dataset:
    """
        Class representing a dataset composed of data items of multiple types,
        all derived from packet data, and providing commonly used functionality
        for handling such datasets, e.g. loading from and saving to secondary
        storage, getting and adding items, shuffling items etc.

        The classes of items that can be stored here are raw packets as well as
        packet projectons.

        Objects of this class can be created with either a preset maximum item
        capacity, in which case, it cannot be changed later, or without such
        capacity. Note that in case of unlimited capacity, there may be more of
        a memory usage overhead, as the used containers for the data items are
        python lists, as opposed to numpy arrays.
    """

    @staticmethod
    def load_dataset(srcdir, name, item_types=None):
        """
            Load a dataset from secondary storage. This function assumes that
            the relevant dataset files are located in the same directory
            (srcdir) and they have the default names which can be constructed
            from the dataset name.

            Parameters
            ----------
            srcdir :    str
                the directory containing all the dataset files
            name :      str
                the dataset name
        """
        configfile = os.path.join(srcdir, '{}_config.ini'.format(name))
        if not os.path.exists(configfile):
            raise IOError('Config file {} does not exist'.format(configfile))
        config = configparser.ConfigParser()
        config.read(configfile)
        general = config['general']
        cap = general['capacity']
        capacity = None if cap == 'None' else int(cap)
        num_data = int(general['num_data'])
        packet_shape = config['packet_shape']
        n_f = int(packet_shape['num_frames'])
        f_h = int(packet_shape['frame_height'])
        f_w = int(packet_shape['frame_width'])
        packet_shape = (n_f, f_h, f_w)
        # load only the data the user requested, else load whatever is present
        # in the dataset
        item_types_sec = config['item_types']
        if item_types is None:
            item_types = {k: (v == 'True') for k, v in item_types_sec.items()}
        data = dict()
        for item_type in ALL_ITEM_TYPES:
            if item_types[item_type] is True:
                filename = os.path.join(srcdir, '{}_{}.npy'.format(
                                        name, item_type))
                data[item_type] = np.load(filename)
            else:
                data[item_type] = None
        targets = np.load(os.path.join(srcdir, '{}_targets.npy'.format(name)))
        metafilename = os.path.join(srcdir, '{}_meta.tsv'.format(name))
        metadata = []
        with open(metafilename) as metafile:
            reader = csv.DictReader(metafile, delimiter='\t')
            for row in reader:
                metadata.append(row)
        dataset = (data, targets, metadata, num_data)
        return numpy_dataset(name, packet_shape, capacity=capacity,
                             dataset=dataset)

    def __init__(self, name, packet_shape, capacity=None, dataset=None,
                 item_types={'yx': False, 'gtux': False, 'gtuy': False,
                             'raw': True}):
        self._name = name
        self._packet_shape = packet_shape
        self._capacity = capacity
        if capacity is None:
            self._appender = self._list_appender
        else:
            self._appender = self._numpy_appender
        if dataset is not None:
            data, targets, metadata, num_data = dataset[0:4]
            self._num_data = num_data
            self._item_types = {k: (True if data[k] is not None else False)
                                for k in ALL_ITEM_TYPES}
        else:
            data = create_data_holders(packet_shape, num_items=capacity,
                                       item_types=item_types)
            targets = np.empty((capacity, 2)) if capacity is not None else []
            metadata = []
            self._num_data = 0
            check_item_types(item_types)
            self._item_types = item_types
        self._data, self._targets = data, targets
        self._metadata = metadata
        self._used_types = tuple(k for k in ALL_ITEM_TYPES
                                 if self._item_types[k] is True)
        self._item_shapes = get_data_item_shapes(packet_shape,
                                                 self._item_types)
        self._metafields = set()

    # helper methods

    def _numpy_appender(self, data, target):
        for key in self._used_types:
            self._data[key][self._num_data] = data[key]
        np.put(self._targets[self._num_data], [0, 1], target)

    def _list_appender(self, data, target):
        for key in self._used_types:
            self._data[key].append(data[key])
        self._targets.append(target)

    # properties

    @property
    def name(self):
        """Name of the dataset."""
        return self._name

    @property
    def capacity(self):
        """Maximum dataset capacity, int (limited) or None (unlimited)."""
        return self._capacity

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
    def item_shapes(self):
        """
            The shape of individual item types in this dataset, in form of a
            dict of str to tuple of int, where the keys are from the module
            constant 'ALL_ITEM_TYPES' and the values represent the shape of
            items as they would be presented from the items themselves by
            getting the numpy.ndarray 'shape' property.
        """
        return self._item_shapes

    # methods

    def get_data_as_arraylike(self, data_slice):
        return tuple(self._data[k][data_slice] for k in self._used_types)

    def get_data_as_dict(self, data_slice):
        return {k: self._data[k][data_slice] for k in self._used_types}

    def get_targets(self, targets_slice):
        return self._targets[targets_slice]

    def get_metadata(self, metadata_slice):
        return self._metadata[metadata_slice]

    def add_data_item(self, packet, target, metadata={}):
        if self._capacity is not None and self._num_data == self._capacity:
            raise Exception('Dataset is already full')
        data_items = convert_packet(packet, item_types=self._item_types)
        self._appender(data_items, target)
        self._metadata.append(metadata)
        self._metafields = self._metafields.union(metadata.keys())
        self._num_data += 1

    def shuffle_dataset(self, num_shuffles):
        """
            Shuffle dataset data, their targets and metadata in unison
            for a given number of times.

            Parameters
            ----------
            num_shuffles :  int
                number of times to shuffle the components of the dataset
        """
        for idx in range(num_shuffles):
            rng_state = np.random.get_state()
            for key in self._used_types:
                np.random.shuffle(self._data[key])
                np.random.set_state(rng_state)
            np.random.shuffle(self._targets)
            np.random.set_state(rng_state)
            np.random.shuffle(self._metadata)

    def save(self, outdir):
        """
            Persist the dataset into secondary storage, with all files stored
            in the same directory (outdir).

            Parameters
            ----------
            outdir :    str
                the directory to store the dataset files in
        """
        if not os.path.exists(outdir):
            raise Exception('Output directory ({}) does not exist!'.format(
                            outdir))
        name = self.name

        # save configuration file
        filename = os.path.join(outdir, '{}_config.ini'.format(name))
        config = configparser.ConfigParser()
        config['general'] = {'capacity': str(self.capacity),
                             'num_data': str(self.num_data)}
        n_f, f_h, f_w = self._packet_shape[0:3]
        config['packet_shape'] = {'frame_height': f_h, 'frame_width': f_w,
                                  'num_frames': n_f}
        config['item_types'] = {k: str(v) for k, v in self.item_types.items()}
        with open(filename, 'w') as configfile:
            config.write(configfile)

        # save data
        for k in self._used_types:
            filename = os.path.join(outdir, '{}_{}.npy'.format(name, k))
            np.save(filename, self._data[k])

        # save targets
        filename = os.path.join(outdir, '{}_targets.npy'.format(name))
        np.save(os.path.join(outdir, filename), self._targets)

        # save metadata
        filename = os.path.join(outdir, '{}_meta.tsv'.format(name))
        with open(filename, 'w') as metafile:
            writer = csv.DictWriter(metafile, fieldnames=self._metafields,
                                    delimiter='\t')
            writer.writeheader()
            writer.writerows(self._metadata)
