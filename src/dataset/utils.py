import abc

import numpy as np


class Shuffler(abc.ABC):

    @abc.abstractmethod
    def shuffle(self, collection):
        """
        Shuffle the passed-in collection in-place.

        :param collection: collection to shuffle
        :param collection: typing.Sequence
        """
        pass

    @abc.abstractmethod
    def keep_state(self):
        """
        Keep the current state of this shuffler as the default state.
        """
        pass

    @abc.abstractmethod
    def reset_state(self):
        """
        Reset state of this shuffler to the current default state.

        When shuffling multiple collections with the same size, if this method
        is called before each collection is shuffled, the result should be that
        all collections should have been shuffled identically, i.e. item
        associations across the collections via corresponding indexes are
        preserved.

        E.g. if

        a = [6, 1, 4, 8, 7, 2, 0, 3, 5 ]
        b = [1, 1, 2, 2, 3, 3, 4, 4, 0 ]

        then after performing:

        shuffler.shuffle(a)
        shuffler.reset_state()
        shuffler.shuffle(b)

        indexes of items are preserved, for example

        a = [3, 7, 0, 4, 8, 1, 5, 2, 6 ]
        b = [4, 3, 4, 2, 2, 1, 0, 3, 1 ]

        State reset should be idempotent, meaning that calling this method
        multiple times should have the same effect as calling it a single time.
        """
        pass


class NumpyRandomShuffler(Shuffler):

    def __init__(self):
        self._state = np.random.get_state()

    def shuffle(self, collection):
        np.random.shuffle(collection)

    def keep_state(self):
        self._state = np.random.get_state()

    def reset_state(self):
        np.random.set_state(self._state)
