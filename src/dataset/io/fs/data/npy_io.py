import os

import numpy as np

import dataset.constants as cons
import dataset.data_utils as dat
import dataset.io.fs.base as fs_io_base


class NumpyDataPersistencyHandler(fs_io_base.FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_DATA_FILES_SUFFIXES = {k: '_{}'.format(k)
                                   for k in cons.ALL_ITEM_TYPES}

    def __init__(self, load_dir=None, save_dir=None, data_files_suffixes={}):
        super(self.__class__, self).__init__(load_dir, save_dir)
        self._data = {}
        for k in cons.ALL_ITEM_TYPES:
            self._data[k] = data_files_suffixes.get(
                k, self.DEFAULT_DATA_FILES_SUFFIXES[k])

    def load_data(self, name, item_types):
        """
            Load dataset data from secondary storage as a dictionary of string
            to numpy.ndarray.

            If a particular item type is not present or should not be loaded,
            it is substituted with an empty list.

            Parameters
            ----------
            :param name:        the dataset name/data filenames prefix.
            :type name:         str
            :param item_types:  types of dataset items to load.
            :type item_types:   typing.Mapping[str, bool]
        """
        self._check_before_read()
        dat.check_item_types(item_types)
        data = {}
        for item_type in cons.ALL_ITEM_TYPES:
            if item_types[item_type]:
                filename = os.path.join(self.loaddir, '{}{}.npy'.format(
                    name, self._data[item_type]))
                data[item_type] = np.load(filename)
            else:
                data[item_type] = []
        return data

    def save_data(self, name, data_items_dict, dtype=np.uint8):
        """
            Persist the dataset data into secondary storage as a set of npy
            files with a common prefix (the dataset name) stored in outdir.

            Parameters
            ----------
            :param name:        the dataset name.
            :type name:         str
            :param data_items_dict: data items to save/persist.
            :type data_items_dict:  typing.Mapping[
                                        str, typing.Sequence[numpy.ndarray]]
            :param dtype:           data type of all items.
            :type data_items_dict:  str or numpy.dtype
        """
        self._check_before_write()
        savefiles = {}
        # save data
        keys = set(cons.ALL_ITEM_TYPES).intersection(data_items_dict.keys())
        for k in keys:
            filename = os.path.join(self.savedir, '{}{}.npy'.format(
                name, self._data[k]))
            data = np.array(data_items_dict[k], dtype=dtype)
            np.save(filename, data)
            savefiles[k] = filename
        return savefiles
