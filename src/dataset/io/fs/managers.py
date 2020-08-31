import abc
import os
import typing as t

import dataset.io.fs.facades as facades
import dataset.io.fs.utils as utils


class FsPersistenceManager(abc.ABC):
    """Base class defining the interface to use for managing persistent items
    for a dataset subcategory (e.g. data, targets, metadata).

    :param io_facades: Mapping from name of the subcategory storage backend to
                       the relevant facade for performing the actual IO for it.
    :param filename_formatters: (optional) Custom mapping from name of specific
                                base filename format for the subcategory items
                                files to the formatter which resolves it based
                                on dataset name and item type. If not set, will
                                use the default formatters as specified in
                                utils.FILENAME_FORMATTERS. Custom formatters
                                take precedence over defaults and the final
                                mapping is made by merging.
    """

    def __init__(
            self,
            io_facades: t.Mapping[str, facades.BaseFilesystemPersistenceFacade],
            filename_formatters: t.Mapping[str, utils.FilenameFormatter] = None
    ):
        if not io_facades:
            # null or an empty dict
            raise ValueError('io_handlers must be initialized')

        custom_formatters = dict(filename_formatters or {})
        _formatters = utils.FILENAME_FORMATTERS.copy()
        _formatters.update(custom_formatters)

        self._formatters = _formatters
        self._facades = io_facades

    # public API

    @abc.abstractmethod
    def load(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            load_types: t.Iterable[str] = None
    ):
        """Load items for the dataset sub-category from secondary storage.

        :param dataset_name: name of the dataset
        :param files_dir: directory storing the top-level dataset config (items
                          are loaded from locations relative to it)
        :param config: subcategory config
        :param load_types: (optional) subset of available item types/fields to
                           load (others will be ignored)
        """
        pass

    @abc.abstractmethod
    def save(self,
             dataset_name: str,
             files_dir: str,
             config: t.Mapping[str, t.Any],
             items: t.Union[t.Mapping[str, t.Sequence], t.Sequence]):
        """Save items for the dataset sub-category to secondary storage.

        :param dataset_name: name of the dataset
        :param files_dir: directory storing the top-level dataset config (items
                          are stored to locations relative to it)
        :param config: subcategory config
        :param items: subcategory items to store
        """
        pass

    @abc.abstractmethod
    def append(self,
               dataset_name: str,
               files_dir: str,
               config: t.Mapping[str, t.Any],
               items: t.Union[t.Mapping[str, t.Sequence], t.Sequence]):
        """Append passed items for the dataset sub-category to already existing
         stored items in secondary storage.

        :param dataset_name: name of the dataset
        :param files_dir: directory storing the top-level dataset config (items
                          to update are stored in locations relative to it)
        :param config: subcategory config
        :param items: subcategory items to append to existing stored items
        """
        pass

    @abc.abstractmethod
    def delete(self,
               dataset_name: str,
               files_dir: str,
               config: t.Mapping[str, t.Any],
               delete_types: t.Iterable[str] = None):
        """Delete items for the dataset sub-category from secondary storage.

        :param dataset_name: name of the dataset
        :param files_dir: directory storing the top-level dataset config (items
                          are stored to locations relative to it)
        :param config: subcategory config
        :param delete_types: (optional) subset of available item types/fields
                             to be deleted (others will be ignored/kept)
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


class SingleFilePerItemTypePersistenceManager(FsPersistenceManager, abc.ABC):

    # public API

    def load(self, dataset_name, files_dir, config, load_types=None):

        # get directory storing the data files
        load_dir = self._check_and_get_files_dir(files_dir)

        # get types of data items to load
        types_config = dict(config['types'])
        _load_types = self._check_and_get_types_subset(types_config,
                                                       load_types)

        # get number of data items (of each type) to load
        num_items = int(config['num_items'])

        # get filenames
        files = self._get_files(dataset_name, load_dir, _load_types, config)

        # load items
        items = {}
        backend_config = config['backend']
        handler = self._facades[backend_config['name']]
        for item_type, type_config in _load_types.items():
            _dtype = type_config['dtype']
            _shape = type_config['shape']
            _file = files[item_type]
            item = handler.load(_file, dtype=_dtype, num_items=num_items,
                                item_shape=_shape)
            items[item_type] = item
        return items

    def save(self, dataset_name, files_dir, config, items):

        # get directory to store the data files
        save_dir = self._check_and_get_files_dir(files_dir)

        # check types config matches passed data
        types_config = dict(config['types'])
        self._check_types_before_update(types_config, items)

        # get filenames
        files = self._get_files(dataset_name, save_dir, types_config, config)

        # save all items
        backend_config = config['backend']
        handler = self._facades[backend_config['name']]
        for item_type, config in types_config.items():
            _file = files[item_type]
            _arr = items[item_type]
            handler.save(_file, _arr)
        return files

    def append(self, dataset_name, files_dir, config, items):

        # get directory to store the data files
        save_dir = self._check_and_get_files_dir(files_dir)

        # check types config matches passed data
        types_config = dict(config['types'])
        self._check_types_before_update(types_config, items)

        # get number of data items (of each type) to save and check against
        # the actual passed data
        num_items = self._check_and_get_num_items(items, config)

        # get filenames
        files = self._get_files(dataset_name, save_dir, types_config, config)

        # append items to existing files
        backend_config = config['backend']
        handler = self._facades[backend_config['name']]
        for item_type, config in types_config.items():
            _file = files[item_type]
            _arr = items[item_type]
            _dtype = config['dtype']
            _shape = config['shape']
            handler.append(_file, _arr, num_items=num_items, dtype=_dtype,
                           item_shape=_shape)
        return files

    def delete(self, dataset_name, files_dir, config, delete_types=None):

        types_config = dict(config['types'])

        _files_dir = self._check_and_get_files_dir(files_dir)
        _itypes = self._check_and_get_types_subset(types_config, delete_types)
        _files = self._get_files(dataset_name, _files_dir, _itypes, config)

        backend_config = config['backend']
        handler = self._facades[backend_config['name']]
        for _filename in _files.values():
            handler.delete(_filename)
        return _files

    # misc. (helper methods)

    def _get_files(self, dataset_name, files_dir, item_types, config):
        # mandatory backend properties/args
        backend_config = config['backend']
        filename_format = backend_config['filename_format']
        filename_extension = backend_config['filename_extension']
        # optional backend properties/args
        suffix = backend_config.get('suffix', None)
        delimiter = backend_config.get('delimiter', None)

        formatter = self._formatters[filename_format]
        create_filename = formatter.create_filename
        append_extension = utils.append_file_extension
        to_full_path = utils.create_full_path

        def _get_filename(item_type):
            _file = create_filename(dataset_name=dataset_name,
                                    item_type=item_type,
                                    suffix=suffix,
                                    delimiter=delimiter)
            _file = append_extension(_file, filename_extension)
            _file = to_full_path(_file, files_dir)
            return _file

        filenames = {item_type: _get_filename(item_type)
                     for item_type in item_types}
        return filenames

    def _check_and_get_types_subset(self, types_config, types_subset):
        available_types = set(types_config.keys())
        if types_subset is not None:
            # user can specify to load only a subset of available types
            _subset = set(types_subset)
        else:
            # otherwise, we default to loading all available item types
            _subset = available_types
        if not _subset.issubset(available_types):
            raise ValueError(f"{_subset} is not a subset of all available "
                             f"types: {available_types}")
        return {itype: types_config[itype] for itype in _subset}

    @staticmethod
    def _check_types_before_update(types_config, items):
        exp_types = set(types_config.keys())
        passed_types = set(items.keys())
        if not passed_types == exp_types:
            _extra = passed_types.difference(exp_types)
            _missing = exp_types.difference(passed_types)
            raise ValueError(f'Mismatch of type config and passed item types:'
                             f'{os.linesep}extra types: {_extra}'
                             f'{os.linesep}missing types: {_missing}')

    @staticmethod
    def _check_and_get_num_items(data, config):
        num_items = int(config['num_items'])
        mismatched_types = {itype: len(data) for itype, data in data.items()
                            if len(data) != num_items}
        if mismatched_types:
            raise ValueError(f'Mismatched number of data items to save, '
                             f'expected: {num_items} for each type, '
                             f'but was: {mismatched_types}')
        return num_items
