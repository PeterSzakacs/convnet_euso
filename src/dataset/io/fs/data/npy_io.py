import os

import numpy as np

import dataset.constants as cons
import dataset.data_utils as dat
import dataset.io.fs.base as fs_io_base


class NumpyDataPersistencyHandler(fs_io_base.FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_DATA_FILES_SUFFIXES = {k: '_{}'.format(k)
                                   for k in cons.ALL_ITEM_TYPES}

    def __init__(self, load_dir=None, save_dir=None, data_file_suffixes=None):
        super(self.__class__, self).__init__(load_dir, save_dir)
        suffixes = {}
        data_file_suffixes = data_file_suffixes or {}
        default_suffixes = self.DEFAULT_DATA_FILES_SUFFIXES
        for k in cons.ALL_ITEM_TYPES:
            suffixes[k] = data_file_suffixes.get(k, default_suffixes[k])
        self._suffixes = suffixes

    def load_data(self, name, item_types, **data_config):
        """
            Load dataset data from secondary storage as a dictionary of string
            to numpy.ndarray.

            Data config is a dict-like structure with additional information
            that can be used during loading.

            Parameters
            ----------
            :param name:        the common prefix of all data file names
            :type name:         str
            :param item_types:  types of items to load.
            :type item_types:   iterable of str
            :param data_config: additional information for loading
            :type data_config:  dict of str,any
        """
        self._check_before_read()
        loaddir = self.loaddir
        _types = dict.fromkeys(item_types, True)
        dat.check_item_types(_types)

        files = self._get_files(name, loaddir, item_types)
        load = np.load
        data = {}
        for item_type in _types:
            data[item_type] = load(files[item_type])
        return data

    def save_data(self, name, items_dict, **data_config):
        """
            Persist the dataset data into secondary storage as a set of raw
            binary files with the contents their respective numpy ndarrays
            (with common filename prefix "name").

            Items in items_dict can be either numpy ndarrays or memmaps.

            Parameters
            ----------
            :param name:        common prefix of name for data files
            :type name:         str
            :param items_dict:  data items to save/persist
            :type items_dict:   dict of str,numpy.ndarray
            :param data_config: additional information for saving
            :type data_config:  dict of str,any
        """
        self._check_before_write()
        savedir = self.savedir
        item_types = dict.fromkeys(items_dict.keys(), True)
        dat.check_item_types(item_types)

        files = self._get_files(name, savedir, item_types)
        save = np.save
        for item_type, items in items_dict.items():
            filename = files[item_type]
            save(filename, items)
        return files

    def append_data(self, name, items_dict, **data_config):
        """
            Append values in items_dict to existing data on secondary storage
            stored in files with the common prefix "name".

            Parameters
            ----------
            :param name:        the dataset name
            :type name:         str
            :param items_dict:  data items to append to dataset (indexed by
                                their item type)
            :type items_dict:   dict of str,numpy.ndarray
            :param data_config: additional information for appending
            :type data_config:  dict of str,any
        """
        self._check_before_write()
        savedir = self.savedir

        item_types = dict.fromkeys(items_dict.keys(), True)
        dat.check_item_types(item_types)
        files = self._get_files(name, savedir, item_types.keys())

        load, save, append = np.load, np.save, np.append
        for item_type in item_types.keys():
            items = items_dict[item_type]
            filepath = files[item_type]
            orig = load(filepath, mmap_mode='r')
            arr = append(orig, items, axis=0)
            save(filepath, arr)
        return files

    def delete_data(self, name, item_types, **data_config):
        """
            Delete dataset data stored on secondary storage in files with a
            common prefix "name".

            Data files are looked up in the 'savedir' attribute.

            :param name:        common prefix of name for data files
            :type name:         str
            :param item_types:  types of items to delete
            :type item_types:   iterable of str
            :param data_config: additional information for deleting
            :type data_config:  dict of str,any
        """
        self._check_before_write()
        savedir = self.savedir
        files = self._get_files(name, savedir, item_types)

        remove = os.remove
        for filename in files.values():
            remove(filename)

    # helper methods

    def _get_files(self, name, directory, item_types):
        suffixes, path_join = self._suffixes, os.path.join
        return {itype: path_join(directory, f"{name}{suffixes[itype]}.npy")
                for itype in item_types}
