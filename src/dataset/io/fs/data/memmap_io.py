import os
import shutil

import numpy as np

import dataset.data.constants as cons
import dataset.data.shapes as d_shapes
import dataset.data.utils as dat
import dataset.io.fs.base as fs_io_base


class MemmapDataPersistencyHandler(fs_io_base.FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_DATA_FILES_SUFFIXES = {k: '_{}'.format(k)
                                   for k in cons.ALL_ITEM_TYPES}

    # constructor

    def __init__(self, load_dir=None, save_dir=None, data_file_suffixes=None):
        super(self.__class__, self).__init__(load_dir, save_dir)
        self._suffixes = {}
        suffixes = data_file_suffixes or {}
        default_suffixes = self.DEFAULT_DATA_FILES_SUFFIXES
        for k in cons.ALL_ITEM_TYPES:
            self._suffixes[k] = suffixes.get(k, default_suffixes[k])

    # methods

    def load_data(self, name, data_config):
        """
            Load dataset data from secondary storage as a dictionary of string
            to numpy.memmap.

            Data config is a dict-like structure with additional information
            needed for properly appending data.

            Mandatory keys and their values in data_config:
            dtype -> numpy.dtype or str (type of stored items)
            num_data -> int or str (number of stored items)
            packet_shape -> (int,int,int) (shape of the original packet)
            item_types -> iterable of str (types of items to load)

            Parameters
            ----------
            :param name:         the common prefix of all data file names
            :type name:          str
            :param data_config:  additional information for loading
            :type data_config:   dict of str,any
        """
        self._check_before_read()
        loaddir = self.loaddir

        dtype = data_config['dtype']
        packet_shape = data_config['packet_shape']
        num_data = int(data_config['num_data'])
        item_types = set(data_config['item_types'])
        _types = dict.fromkeys(item_types, True)

        dat.check_item_types(_types)
        item_shapes = d_shapes.get_data_item_shapes(packet_shape, _types)
        files = self._get_files(name, loaddir, _types)

        data = {}
        for item_type in _types:
            filepath = files[item_type]
            shape = item_shapes[item_type]
            mmap = np.memmap(filepath, dtype=dtype, mode='r',
                             shape=(num_data, *shape))
            data[item_type] = mmap
        return data

    def save_data(self, name, items_dict):
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
        """
        self._check_before_write()
        suffixes = self._suffixes
        savedir = self.savedir

        path_join, copy = os.path.join, shutil.copy
        memmap, ndarray = np.memmap, np.ndarray
        # memmap is a subclass of ndarray, so just checking that is enough
        invalid_items = {item_type: type(item)
                         for item_type, item in items_dict.items()
                         if not isinstance(item, ndarray)}
        if invalid_items:
            raise ValueError(f"Unsupported types of items passed (allowed: "
                             f"numpy.ndarray, numpy.memmap): {invalid_items}")

        item_types = items_dict.keys()
        files = self._get_files(name, savedir, item_types)
        savefiles = {}
        for item_type in item_types:
            item, filepath = items_dict[item_type], files[item_type]
            if isinstance(item, memmap):
                copy(item.filename, filepath)
            else:
                mmap = memmap(filepath, dtype=item.dtype, mode='w+',
                              shape=item.shape)
                mmap[:] = item[:]
            savefiles[item_type] = filepath
        return savefiles

    def save_empty(self, name, item_types):
        """
            Create placeholder empty files for dataset data on secondary
            storage.

            This method is primarily intended for workflows, which attempt to
            create dataset data gradually instead of saving all at once.

            :param name:        common prefix of name for new data files
            :type name:         str
            :param item_types:  types of items for which files should be
                                created
            :type item_types:   iterable of str
        """
        self._check_before_write()
        savedir = self.savedir
        _types = set(item_types)
        files = self._get_files(name, savedir, _types)
        # files = {itype: path_join(savedir, f"{name}{suffixes[itype]}.memmap")
        #          for itype in _types}
        for filepath in files.values():
            open(filepath, mode='w').close()
        return files

    def append_data(self, name, items_dict, data_config):
        """
            Append values in items_dict to existing data on secondary storage
            stored in files with the common prefix "name".

            Data config is a dict-like structure with additional information
            needed for properly appending data.

            Mandatory keys and their values in data_config:
            dtype -> numpy.dtype or str (type of stored items)
            num_data -> int or str (number of stored items)
            packet_shape -> (int,int,int) (shape of the original packet)

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
        dtype = data_config['dtype']
        num_data = data_config['num_data']
        packet_shape = data_config['packet_shape']

        item_types = dict.fromkeys(items_dict.keys(), True)
        files = self._get_files(name, savedir, item_types.keys())
        item_shapes = d_shapes.get_data_item_shapes(packet_shape, item_types)

        memmap = np.memmap
        savefiles = {}
        for item_type in item_types.keys():
            filepath = files[item_type]
            items, shape = items_dict[item_type], item_shapes[item_type]
            mmap_shape = (num_data + len(items), *shape)
            mmap = memmap(filepath, dtype=dtype, shape=mmap_shape, mode='r+')
            mmap[num_data:, :] = items
            savefiles[item_type] = filepath
        return savefiles

    def delete_data(self, name, item_types):
        """
            Delete dataset data stored on secondary storage in files with a common
            prefix "name".

            Data files are looked up in the 'savedir' attribute.

            :param name:        common prefix of name for data files
            :type name:         str
            :param item_types:  types of items to delete
            :type item_types:   iterable of str
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
        return {itype: path_join(directory, f"{name}{suffixes[itype]}.memmap")
                for itype in item_types}
