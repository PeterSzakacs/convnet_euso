import dataset.targets.constants as cons
import dataset.targets.utils as utils


class TargetsFacade:
    """
    Facade class providing a container for expected outputs (targets) to use
    for training/evaluating a trained model and some additional operations
    on these targets.

    :param targets_dict: targets to store in this facade instance
    :type targets_dict: dict of str,numpy.ndarray
    :param copy_on_read: optional flag indicating if item retrieval should
                         return a copy of the stored targets instead of the
                         original (default: False)
    :type copy_on_read: bool
    """

    def __init__(self, targets_dict, copy_on_read=False):
        targets = {k: v for k, v in targets_dict.items() if v is not None}
        target_types = {k: True for k, v in targets.items()}

        utils.check_target_types(target_types)
        utils.check_targets_length(targets)

        self._used_types = tuple(k for k in cons.TARGET_TYPES
                                 if target_types.get(k, False) is True)
        self._target_types = target_types
        self._targets = targets
        self._num_items = len(list(targets.values())[0])
        self.copy_on_read = copy_on_read

    def __len__(self):
        return self._num_items

    # properties

    @property
    def target_types(self):
        """
            The type of targets in this dataset, as a dict of str to bool, in
            which all keys are from the 'TARGET_TYPES' module constant and
            the values represent whether a collection of targets of this type
            is present in this dataset.
        """
        return self._target_types

    # methods

    def get_targets_as_tuple(self, idx=None):
        """Return targets contained in this facade instance as a tuple of
        numpy.ndarrays ordered by their type as defined in TARGET_TYPES.

        :param idx: optional indexing parameter
        :type idx: None or int or range or Sequence of int
        :return: tuple of numpy.ndarray
        """
        data, idxs = self._targets, self._get_indexes(idx)
        if self.copy_on_read:
            return tuple(data[k][idxs].copy() for k in self._used_types)
        else:
            return tuple(data[k][idxs] for k in self._used_types)

    def get_targets_as_dict(self, idx=None):
        """Return targets contained in this facade instance as a dict of
        str to numpy.ndarray.

        Note that only keys for item types contained in this facade have
        defined values, keys for other item types are not even present.

        :param idx: optional indexing parameter
        :type idx: None or int or range or Sequence of int
        :return: dict of (str,numpy.ndarray)
        """
        data, idxs = self._targets, self._get_indexes(idx)
        if self.copy_on_read:
            return {k: data[k][idxs].copy() for k in self._used_types}
        else:
            return {k: data[k][idxs] for k in self._used_types}

    def shuffle(self, shuffler, shuffler_state_resetter):
        for target_type in self._used_types:
            items = self._targets[target_type]
            shuffler(items)
            shuffler_state_resetter()

    # helper methods

    def _get_indexes(self, indexing_obj):
        if indexing_obj is None:
            return range(self._num_items)
        else:
            return indexing_obj
