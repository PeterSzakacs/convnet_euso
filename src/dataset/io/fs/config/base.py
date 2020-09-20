import abc
import typing as t


class SingleFileConfigPersistenceManager(abc.ABC):
    """
    Base class/interface for all file-backed config managers defining the
    contract for all implementations.

    The manager implementation also implicitly converts configs to and from
    their internal representation (in the form of a Mapping/dict) that is
    to be used both for loading the actual items and when creating a dataset
    object instance (incl. additional attributes).

    For all implementations, the following constraint(s) shall apply:

    - 1.) The config as stored on the filesystem shall be referenced using a
    single file (the "master config") which is to be passed to all methods
    - 2.) Constraint 1. does not imply that the config properties and values
    cannot be stored across multiple files for example for readability by
    humans (e.g. per-dataset section, per-item type etc.). It just says that
    there should be a single master config which serves also as an entry point
    to the others (via references/links/etc. within it).
    - 3.) The master config shall reside at the top of the directory subtree
    containing the actual item-holding files of the dataset (and other files
    possibly holding parts of the config - see constraint 2.)
    """

    @abc.abstractmethod
    def load(
            self,
            file: str
    ) -> t.Mapping[str, t.Any]:
        """Load dataset attributes/properties from the given master config.

        :param file: name/path of the master config file
        :return: representation of dataset attributes/properties as a dict or
                 mapping
        """
        pass

    @abc.abstractmethod
    def save(
            self,
            file: str,
            dataset_attrs: t.Mapping[str, t.Any]
    ):
        """Save dataset attributes/properties to a config rooted at the given
        master config.

        :param file: name/path of the master config file
        :param dataset_attrs: dataset attributes/properties to persist
        """
        pass
