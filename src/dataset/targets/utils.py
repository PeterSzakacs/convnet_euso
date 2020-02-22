import functools
import operator

import dataset.targets.constants as cons


def check_target_types(target_types):
    """
        Utility function to check if the passed in argument is a valid flag for
        indicating supported/requested types of dataset targets.

        A valid flag is defined as a dict of str to bool, whose keys are solely
        from the TARGET_TYPES constant/enum and at least one of these keys
        maps to a value of True.

        :param target_types: flag to be checked for validity
        :type target_types: dict of (str,bool)
        :raises ValueError: if the passed-in flag is not valid
    """
    # all keys are set to false
    if not functools.reduce(operator.or_, target_types.values(), False):
        raise ValueError(('At least one item type (possible types: {})'
                         ' must be used in the dataset').format(
                         cons.TARGET_TYPES))
    illegal_keys = target_types.keys() - set(cons.TARGET_TYPES)
    if len(illegal_keys) > 0:
        raise ValueError(('Unknown keys found: {}'.format(illegal_keys)))


def check_targets_length(targets_dict):
    """
        Utility function to check if all targets in the passed dict are
        of the same length.

        :param targets_dict: dict of targets to be checked for validity
        :type targets_dict: dict of str,collections.abc.Sized
        :raises ValueError: if the passed-in dict is not valid
    """
    targets_list = [targets for targets in targets_dict.values()]
    n_targets = len(targets_list[0])
    for targets_type, targets in targets_dict.items():
        expected, actual = n_targets, len(targets)
        if expected != actual:
            raise ValueError("Passed collection of targets for type '{}' "
                             "does not have the same length as those of "
                             "previously added types. Expected: {}, got: "
                             "{}".format(targets_type, expected, actual))
