import abc
import typing as t


class BaseFilesystemPersistenceFacade(abc.ABC):
    """
    Basic facade interface to be implemented by classes performing IO on items
    of a dataset section.
    """

    @abc.abstractmethod
    def load(
            self, *args, **kwargs
    ) -> t.Union[t.Mapping[str, t.Sequence], t.Sequence]:
        """
        Load items for dataset section.

        This method can return either a sequence representing data of a given
        item type or a mapping/dict of such sequences indexed by the item type
        name as the mapping key (if a single call supports loading multiple
        types).

        NOTE 1: Whether the stored items are actually loaded into main memory
        shall be left as an implementation detail. If not loaded, the returned
        sequence(s) provide a view-like object to access the data kept on disk.

        NOTE 2: If multiple item types can be loaded in a single call, the same
        implementing class can support loading of only a subset of all stored
        types.
        """
        pass

    @abc.abstractmethod
    def save(self, *args, **kwargs) -> None:
        """
        Save new items for dataset section.

        The items passed in have to be in the same or similar format as that
        returned by the corresponding load() method on the implementing class.

        If multiple types can be returned from load(), then this method must
        also be able to support saving of multiple types.
        """
        pass

    @abc.abstractmethod
    def append(self, *args, **kwargs) -> None:
        """
        Append new items to already stored items for dataset section.

        The items passed in have to be in the same or similar format as that
        returned by the corresponding load() method on the implementing class.

        If multiple types can be returned from load(), then this method must
        also be able to support appending items of multiple types to the
        corresponding stored items.
        """
        pass

    @abc.abstractmethod
    def delete(self, *args, **kwargs) -> None:
        """
        Delete items for dataset section.

        This method shall not require any actual items to be passed to it for
        the purpose of deletion, only whatever is needed by the implementation
        to remove the stored items (usually just filename(s)).

        NOTE: If the implementation supports both loading multiple item types
        from a single call to load() as well as loading only a subset of stored
        types, then this method must also enable the option to only partially
        delete a section (i.e. delete only items of a specific type).
        """
        pass
