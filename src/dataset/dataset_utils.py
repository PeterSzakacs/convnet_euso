import numpy as np

import dataset.data_utils as dat
import dataset.metadata_utils as meta
import dataset.target_utils as targ


class NumpyDataset:
    """
        Class representing a dataset composed of data items of multiple types,
        all derived from packet data, and providing commonly used functionality
        for handling such datasets, e.g. getting and adding items, shuffling
        items etc.

        The types of items that can be stored here are raw packets as well as
        packet projectons.
    """

    def __init__(self, name, packet_shape, resizable=True, dtype=np.uint8,
                 item_types=None, **attrs):
        item_types = item_types or {'raw': True, 'gtux': False, 'gtuy': False,
                                    'yx': False}
        self._data = dat.DataHolder(packet_shape, item_types=item_types,
                                    dtype=dtype)
        self._targ = targ.TargetsHolder()
        self._meta = meta.MetadataHolder()
        self._num_data = 0
        self._attrs = attrs
        self.resizable = resizable
        self.name = name

    def __str__(self):
        attrs_dict = {
            'name': self.name,
            'packet_shape': self.accepted_packet_shape,
            'item_types': self.item_types,
            'dtype': self.dtype}
        return str(attrs_dict)

    # helper methods

    def _get_items_slice(self, items_slice_or_idx=None):
        return (slice(None) if items_slice_or_idx is None
                else items_slice_or_idx)

    # properties

    @property
    def attributes(self):
        item_types, dtype = self.item_types, self.dtype
        shapes = self.item_shapes
        types = {it: {'dtype': dtype, 'shape': shapes[it]}
                 for it, is_present in item_types.items() if is_present}
        return {
            'name': self.name,
            'num_items': self.num_data,
            'data': {
                'packet_shape': self.accepted_packet_shape,
                'backend': {
                    'name': 'npy',
                    'filename_format': 'name_with_type_suffix',
                    'filename_extension': 'npy',
                },
                'types': types,
            },
            'targets': {
                'backend': {
                    'name': 'npy',
                    'filename_format': 'name_with_suffix',
                    'filename_extension': 'npy',
                    'suffix': 'class_targets',
                },
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8',
                        'shape': (2, ),
                    }
                },
            },
            'metadata': self._meta.attributes
        }

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
    def dtype(self):
        """Datatype of dataset items."""
        return self._data.dtype

    @dtype.setter
    def dtype(self, value):
        """Datatype of dataset items."""
        self._data.dtype = value

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
        return self._data.item_types

    @property
    def accepted_packet_shape(self):
        """
            The shape of accepted raw packets to be converted to dataset items.
        """
        return self._data.accepted_packet_shape

    @property
    def item_shapes(self):
        """
            The shape of individual item types in this dataset, in form of a
            dict of str to tuple of int, where the keys are from the module
            constant 'ALL_ITEM_TYPES' and the values represent the shape of
            items as they would be presented from the items themselves by
            getting the numpy.ndarray 'shape' property.
        """
        return self._data.item_shapes

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
        return self._meta.metadata_fields

    # add/get items

    def get_data_as_arraylike(self, data_slice_or_idx=None):
        return self._data.get_data_as_arraylike(data_slice_or_idx)

    def get_data_as_dict(self, data_slice_or_idx=None):
        return self._data.get_data_as_dict(data_slice_or_idx)

    def get_targets(self, targets_slice_or_idx=None):
        return self._targ.get_targets_as_arraylike(targets_slice_or_idx)[0]

    def get_metadata(self, metadata_slice_or_idx=None):
        s = self._get_items_slice(metadata_slice_or_idx)
        return self._meta[s]

    def add_data_item(self, packet, target, metadata={}):
        if not self._resizable:
            raise Exception('Cannot add items to dataset')
        self._data.append_packet(packet)
        self._targ.append({'classification': target})
        self._meta.append(metadata)
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
        shuffler = np.random.shuffle
        for idx in range(num_shuffles):
            rng_state = np.random.get_state()
            state_resetter = lambda: np.random.set_state(rng_state)
            self._data.shuffle(shuffler, state_resetter)
            self._targ.shuffle(shuffler, state_resetter)
            self._meta.shuffle(shuffler)

    def is_compatible_with(self, other_dataset, check_dtype=False):
        """
            Check if the other dataset is compatible with the current dataset.

            Essentially checks if these datasets have the same types of items
            and accept the same shape of packets.

            Parameters
            ----------
            :param other_dataset:   the dataset to check compatibility of
            :type other_dataset:    utils.dataset_utils.NumpyDataset
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
            :type other_dataset:    utils.dataset_utils.NumpyDataset
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
        self._data.extend(other_dataset.get_data_as_dict(s))
        self._targ.extend({'classification': other_dataset.get_targets(s)})
        metadata = other_dataset.get_metadata(s)
        self._meta.extend(other_dataset.get_metadata(s))
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
        self._meta.add_metafield(name, default_value=default_value)
