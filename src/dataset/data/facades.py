import dataset.data.constants as cons
import dataset.data.utils as utils


class DataFacade:
    """
    Facade class providing a container for packet data-derived items in a
    tagged dataset and some additional operations on them.

    :param packet_shape: shape of the original packets from which the passed
                         data items were derived, represented as a tuple of
                         (number of frames, frame height, frame width)
    :type packet_shape: tuple of int
    :param items_dict: data items to store in this facade instance
    :type items_dict: dict of str,numpy.ndarray
    :param copy_on_read: optional flag indicating if item retrieval should
                         return a copy of the stored items instead of the
                         original (default: False)
    :type copy_on_read: bool
    """

    def __init__(self, packet_shape, items_dict, copy_on_read=False):
        packet_shape = tuple(packet_shape)
        items = {k: v for k, v in items_dict.items() if v is not None}
        item_types = {k: True for k, v in items.items()}

        utils.check_item_types(item_types)
        utils.check_items_length(items)

        self._used_types = tuple(k for k in cons.ALL_ITEM_TYPES
                                 if item_types.get(k, False) is True)
        self._packet_shape = packet_shape
        self._item_types = item_types
        self._data = items
        self._num_items = len(list(items.values())[0])
        self.copy_on_read = copy_on_read

    def __len__(self):
        return self._num_items

    # properties

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
    def packet_shape(self):
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
        return {k: item.shape[1:] for k, item in self._data.items()}

    # methods

    def get_data_as_tuple(self, idx=None):
        """Return data items contained in this facade instance as a tuple of
        numpy.ndarrays ordered by their type as defined in ALL_ITEM_TYPES.

        :param idx: optional indexing parameter
        :type idx: None or int or range or Sequence of int
        :return: tuple of numpy.ndarray
        """
        data, idxs = self._data, self._get_indexes(idx)
        if self.copy_on_read:
            return tuple(data[k][idxs].copy() for k in self._used_types)
        else:
            return tuple(data[k][idxs] for k in self._used_types)

    def get_data_as_dict(self, idx=None):
        """Return data items contained in this facade instance as a dict of
        str to numpy.ndarray.

        Note that only keys for item types contained in this facade have
        defined values, keys for other item types are not even present.

        :param idx: optional indexing parameter
        :type idx: None or int or range or Sequence of int
        :return: dict of (str,numpy.ndarray)
        """
        data, idxs = self._data, self._get_indexes(idx)
        if self.copy_on_read:
            return {k: data[k][idxs].copy() for k in self._used_types}
        else:
            return {k: data[k][idxs] for k in self._used_types}

    def shuffle(self, shuffler):
        shuffler.reset_state()
        for item_type in self._used_types:
            items = self._data[item_type]
            shuffler.shuffle(items)
            shuffler.reset_state()

    # helper methods

    def _get_indexes(self, indexing_obj):
        if indexing_obj is None:
            return range(self._num_items)
        else:
            return indexing_obj
