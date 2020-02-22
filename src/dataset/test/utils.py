import dataset.utils as utils


class MockShuffler(utils.Shuffler):

    def __init__(self, swap_indexes=None):
        self._indexes = swap_indexes
        self._tmp = swap_indexes

    def keep_state(self):
        self._indexes = self._tmp

    def reset_state(self):
        self._indexes = self._tmp

    def shuffle(self, collection):
        idx1, idx2 = self._indexes[:]
        tmp = collection[idx1]
        collection[idx1] = collection[idx2]
        collection[idx2] = tmp
        # Temporarily set swap indexes to None until state reset is called.
        # This is used to simulate the base Shuffler behavior when the code
        # under test is designed to handle multiple collections which should
        # be shuffled while preserving association. If the code does not call
        # reset_state, at very least an exception is raised.
        self._indexes = None
