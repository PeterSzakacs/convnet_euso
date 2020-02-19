import functools
import operator

import dataset.data.constants as cons


def check_item_types(item_types):
    """
        Utility function to check if the passed in argument is a valid flag for
        indicating supported/requested types of dataset items (all identical to
        or derived from a packet of recorded data).

        A valid flag is defined as a dict of str to bool, whose keys are solely
        from the ALL_ITEM_TYPES constant/enum and at least one of these keys
        maps to a value of True.

        :param item_types: flag to be checked for validity
        :type item_types: dict of (str,bool)
        :raises ValueError: if the passed-in flag is not valid
    """
    # all keys are set to false
    if not functools.reduce(operator.or_, item_types.values(), False):
        raise ValueError(('At least one item type (possible types: {})'
                         ' must be used in the dataset').format(
                         cons.ALL_ITEM_TYPES))
    illegal_keys = item_types.keys() - set(cons.ALL_ITEM_TYPES)
    if len(illegal_keys) > 0:
        raise ValueError(('Unknown keys found: {}'.format(illegal_keys)))


def check_items_length(items_dict):
    """
        Utility function to check if all data items in the passed dict are
        all of the same length.

        :param items_dict: dict of data items to be checked for validity
        :type items_dict: dict of str,collections.abc.Sized
        :raises ValueError: if the passed-in dict is not valid
    """
    items_list = [items for items in items_dict.values()]
    n_items = len(items_list[0])
    for item_type, items in items_dict.items():
        expected, actual = n_items, len(items)
        if expected != actual:
            raise ValueError("Passed collection of items for type '{}' "
                             "does not have the same length as those of "
                             "previously added types. Expected: {}, got: "
                             "{}".format(item_type, expected, actual))
