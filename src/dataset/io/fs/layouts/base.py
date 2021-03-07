import abc
import os
import typing as t

import common.providers as providers
import dataset.io.fs.facades as facades
import dataset.io.fs.utils as utils


class DatasetSectionFileLayoutHandler(abc.ABC):
    """Base class defining a common interface for performing operations on
    FS-backed dataset section items (data, targets, metadata) regardless of
    the layout of files that constitute them.

    Note that a call to any of the methods of this class shall perform a given
    operation only on items of a single section (i.e. not on data and targets
    at once).
    """

    def __init__(
            self,
            io_facades_provider: providers.ClassInstanceProvider = None,
            formatters_provider: providers.ClassInstanceProvider = None
    ):
        """
        :param io_facades_provider: (optional) Custom provider used to resolve
                                    the 'backend' config value to the correct
                                    IO facade to use. If not provided, will use
                                    the default provider as returned by
                                    :module:`dataset.io.fs.facades`.
        :param formatters_provider: (optional) Custom provider used to resolve
                                    the 'filename_format' config value to the
                                    correct FilenameFormatter to use. If not
                                    provided, will use the default provider as
                                    returned by :module:`dataset.io.fs.utils`.
        """

        self._formatters_provider = (formatters_provider
                                     or utils.get_formatters_provider())
        self._facades_provider = (io_facades_provider
                                  or facades.get_facades_provider())

    # public API

    @abc.abstractmethod
    def load(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            load_types: t.Iterable[str] = None
    ):
        """Load items for the dataset section from secondary storage.

        :param dataset_name: name of the dataset.
        :param files_dir: directory storing the top-level dataset config (items
                          are loaded from locations relative to it).
        :param config: section config.
        :param load_types: (optional) subset of available item types/fields to
                           load (others will be ignored).
        """
        pass

    @abc.abstractmethod
    def save(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            items: t.Union[t.Mapping[str, t.Sequence], t.Sequence]
    ):
        """Save items for the dataset section to secondary storage.

        :param dataset_name: name of the dataset.
        :param files_dir: directory storing the top-level dataset config (items
                          are stored to locations relative to it).
        :param config: section config.
        :param items: section items to store.
        """
        pass

    @abc.abstractmethod
    def append(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            items: t.Union[t.Mapping[str, t.Sequence], t.Sequence]
    ):
        """Append passed items for the dataset section to already existing
         stored items in secondary storage.

        :param dataset_name: name of the dataset.
        :param files_dir: directory storing the top-level dataset config (items
                          to update are stored in locations relative to it).
        :param config: section config.
        :param items: section items to append to existing stored items.
        """
        pass

    @abc.abstractmethod
    def delete(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            delete_types: t.Iterable[str] = None
    ):
        """Delete items for the dataset section from secondary storage.

        :param dataset_name: name of the dataset.
        :param files_dir: directory storing the top-level dataset config (items
                          are stored to locations relative to it).
        :param config: section config.
        :param delete_types: (optional) subset of available item types/fields
                             to be deleted (others will be ignored/kept).
        """
        pass

    # misc. (helper methods)

    @staticmethod
    def _check_and_get_files_dir(files_dir):
        if files_dir is None:
            raise ValueError("Files dir can not be null")
        _dir = os.path.abspath(files_dir)
        if not os.path.isdir(_dir):
            raise IOError(f"{files_dir} is not a valid directory")
        return _dir
