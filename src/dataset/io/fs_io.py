import abc
import ast
import configparser
import os

import numpy as np

import dataset.constants as cons
import dataset.data_utils as dat
import dataset.dataset_utils as ds
import dataset.metadata_utils as meta
import utils.io_utils as io_utils

class FsPersistencyHandler(abc.ABC):

    def __init__(self, load_dir=None, save_dir=None):
        super(FsPersistencyHandler, self).__init__()
        self.loaddir = load_dir
        self.savedir = save_dir

    # properties

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, value):
        if value is not None and not os.path.isdir(value):
            raise IOError('Invalid save directory: {}'.format(value))
        self._savedir = value

    @property
    def loaddir(self):
        return self._loaddir

    @loaddir.setter
    def loaddir(self, value):
        if value is not None and not os.path.isdir(value):
            raise IOError('Invalid load directory: {}'.format(value))
        self._loaddir = value

    def _check_before_write(self, err_msg='Save directory not set'):
        if self.savedir is None:
            raise Exception(err_msg)
        else:
            return True

    def _check_before_read(self, err_msg='Load directory not set'):
        if self.loaddir is None:
            raise Exception(err_msg)
        else:
            return True


class DatasetMetadataFsPersistencyHandler(FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_METADATA_FILE_SUFFIX = '_meta'

    def __init__(self, load_dir=None, save_dir=None, metafile_suffix=None):
        super(DatasetMetadataFsPersistencyHandler, self).__init__(
            load_dir, save_dir)
        self._meta = metafile_suffix or self.DEFAULT_METADATA_FILE_SUFFIX

    def load_metadata(self, name, metafields=None):
        """
            Load dataset metadata from secondary storage as a list of dicts.

            Accepts optionally the names of fields to load. The returned list
            of dictionaries will in that case contain these fields as keys,
            with any extra field values unparsed and indexed by the default
            None key.

            Parameters
            ----------
            :param name:        the dataset name/metadata filename prefix.
            :type name:         str
            :param metafields:  (optional) names of fields to load.
            :type metafields:   typing.Iterable[str]
        """
        meta_fields = metafields
        if meta_fields is not None:
            meta_fields = set(metafields)
        filename = os.path.join(self.loaddir, '{}{}.tsv'.format(
            name, self._meta))
        meta = io_utils.load_TSV(filename, selected_columns=meta_fields)
        return meta

    def save_metadata(self, name, metadata, metafields=None,
                              metafields_order=None):
        """
            Persist dataset metadata into secondary storage as a TSV file
            stored in outdir with the given order of fields (columns).

            If metafields is not passed, it is derived by iterating over the
            metadata before saving and finding all unique fieldnames. Passing
            this argument can therefore speed up this method, but the caller
            is responsible for making sure all metadata fields are accounted
            for.

            If no ordering is passed, the fields are sorted by their name. If
            an ordering is passed and does not account for all fields present
            in metafields (regardless if they were derived from metadata or
            passed in explicitly) an exception is raised.

            Parameters
            ----------
            :param name:        the dataset name.
            :type name:         str
            :param metadata:    metadata to save/persist.
            :type metadata:     typing.Sequence[typing.Mapping[
                                    str, typing.Any]]
            :param metafields:  (optional) names of all fields in the metadata.
            :type metafields:   typing.Set[str]
            :param metafields_order:    (optional) ordering of fields (columns)
                                        in the created TSV.
            :type metafields_order:     typing.Sequence[str]
        """
        metafields = metafields or meta.extract_metafields(metadata)
        if metafields_order is not None:
            fields_in_order = set(metafields_order)
            diff = fields_in_order.symmetric_difference(metafields)
            if not not diff:
                raise Exception('Metadata field order contains more or fewer '
                                'fields than are present in metadata.\n'
                                'Metafields in order: {}\nMetafields: {}'
                                .format(fields_in_order, metafields))
        else:
            metafields_order = list(metafields)
            metafields_order.sort()
        # save metadata
        filename = os.path.join(self.savedir, '{}{}.tsv'.format(
            name, self._meta))
        io_utils.save_TSV(filename, metadata, metafields_order,
                          file_exists_overwrite=True)
        return filename


class DatasetTargetsFsPersistencyHandler(FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_CLASSIFICATION_TARGETS_FILE_SUFFIX = '_class_targets'

    def __init__(self, load_dir=None, save_dir=None,
                 classification_targets_file_suffix=None):
        super(DatasetTargetsFsPersistencyHandler, self).__init__(
            load_dir, save_dir)
        self._targ = (classification_targets_file_suffix or
                      self.DEFAULT_CLASSIFICATION_TARGETS_FILE_SUFFIX)

    def load_targets(self, name):
        """
            Load dataset targets from secondary storage as a numpy.ndarray.

            Parameters
            ----------
            :param name:        the dataset name/targets filename prefix.
            :type name:         str
        """
        filename = '{}{}.npy'.format(name, self._targ)
        return np.load(os.path.join(self.loaddir, filename))

    def save_targets(self, name, targets):
        """
            Persist the dataset targets into secondary storage as an npy file
            stored in outdir.

            Parameters
            ----------
            :param name:        the dataset name.
            :type name:         str
            :param targets:     targets to save/persist.
            :type targets:      typing.Sequence[numpy.ndarray]
        """
        # save targets
        filename = os.path.join(self.savedir, '{}{}.npy'.format(
            name, self._targ))
        np.save(filename, targets)
        return filename


class DatasetFsPersistencyHandler(FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_CONFIG_FILE_SUFFIX = '_config'
    DEFAULT_DATA_FILES_SUFFIXES = {k: '_{}'.format(k)
                                   for k in cons.ALL_ITEM_TYPES}

    def __init__(self, load_dir=None, save_dir=None, data_files_suffixes={},
                 configfile_suffix=None, targets_handler=None,
                 metadata_handler=None):
        super(DatasetFsPersistencyHandler, self).__init__(
            load_dir, save_dir)
        self._conf = configfile_suffix or self.DEFAULT_CONFIG_FILE_SUFFIX
        self._data = {}
        for k in cons.ALL_ITEM_TYPES:
            self._data[k] = data_files_suffixes.get(
                k, self.DEFAULT_DATA_FILES_SUFFIXES[k])
        self._target_handler = (targets_handler or
                                DatasetTargetsFsPersistencyHandler(
                                    load_dir=load_dir, save_dir=save_dir))
        self._meta_handler   = (metadata_handler or
                                DatasetMetadataFsPersistencyHandler(
                                    load_dir=load_dir, save_dir=save_dir))

    # properties

    @property
    def targets_persistency_handler(self):
        return self._target_handler

    @property
    def metadata_persistency_handler(self):
        return self._meta_handler

    # dataset load

    def load_dataset_config(self, name):
        # TODO: make preload dataset return an actual empty dataset?
        # what about num data though?
        """
            Loads the configuration of a dataset from secondary storage.

            Essentially, this function creates a dictionary of all 'public'
            attributes and properties of an existing dataset without loading
            its data, targets and metadata into memory.

            Parameters
            ----------
            :param name:        the dataset name/config filename prefix.
            :type name:         str
        """
        self._check_before_read()
        configfile = os.path.join(self.loaddir, '{}{}.ini'.format(
            name, self._conf))
        if not os.path.exists(configfile):
            raise FileNotFoundError('Config file {} does not exist'.format(
                                    configfile))
        config = configparser.ConfigParser()
        config.read(configfile, encoding='UTF-8')
        attrs = {}
        general = config['general']
        attrs['num_data'] = int(general['num_data'])
        attrs['metafields'] = ast.literal_eval(general['metafields'])
        attrs['dtype'] = general['dtype']
        packet_shape = config['packet_shape']
        n_f = int(packet_shape['num_frames'])
        f_h = int(packet_shape['frame_height'])
        f_w = int(packet_shape['frame_width'])
        attrs['packet_shape'] = (n_f, f_h, f_w)
        item_types_sec = config['item_types']
        item_types = {k: (v == 'True') for k, v in item_types_sec.items()}
        attrs['item_types'] = item_types
        return attrs

    def load_empty_dataset(self, name, item_types=None):
        """
            Create a dataset from configuration stored in secondary storage
            without loading any of its actual contents (data, targets,
            metadata).

            Parameters
            ----------
            :param name:        the dataset name/config filename prefix.
            :type name:         str
            :param item_types:  (optional) types of dataset items to load.
            :type item_types:   typing.Mapping[str, bool]
        """
        attrs = self.load_dataset_config(name)
        itypes = item_types or attrs['item_types']
        dataset = ds.NumpyDataset(name, attrs['packet_shape'],
                                  item_types=itypes, dtype=attrs['dtype'])
        return dataset

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

    def load_dataset(self, name, item_types=None):
        """
            Load a dataset from secondary storage.

            This function assumes that the relevant dataset files are located
            in the same directory (loaddir).

            Parameters
            ----------
            :param name:        the dataset name.
            :type name:         str
            :param item_types:  (optional) types of dataset items to load.
            :type item_types:   typing.Mapping[str, bool]
        """
        # TODO: Think of a way to load dataset with items that does not depend
        # on knowledge of NumpyDataset internals
        # currently excluded from unit tests for that very reason
        self._check_before_read()
        config = self.load_dataset_config(name)
        itypes = item_types or config['item_types']
        dataset = ds.NumpyDataset(name, config['packet_shape'],
                                  item_types=itypes)
        data = self.load_data(name, dataset.item_types)
        targets = self._target_handler.load_targets(name)
        metadata = self._meta_handler.load_metadata(name)
        dataset._data.extend(data)
        dataset._targ.extend({'classification': targets})
        dataset._meta.extend(metadata)
        dataset._num_data = config['num_data']
        return dataset

    # dataset save/persist

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

    def save_dataset(self, dataset, metafields_order=None):
        """
            Persist the dataset into secondary storage, with all files stored
            in the same directory (outdir).

            Parameters
            ----------
            :param dataset:        the dataset to save/persist.
            :type dataset:         utils.dataset_utils.NumpyDataset
            :param metafields_order:    (optional) ordering of fields (columns)
                                        in the created metadata TSV.
            :type metafields_order:     typing.Sequence[str]
        """
        self._check_before_write()
        name = dataset.name
        metadata, metafields = dataset.get_metadata(), dataset.metadata_fields
        self._meta_handler.save_metadata(name, metadata, metafields=metafields,
                                         metafields_order=metafields_order)
        targets = dataset.get_targets()
        self._target_handler.save_targets(name, targets)
        data = dataset.get_data_as_dict()
        self.save_data(name, data, dtype=dataset.dtype)

        # save configuration file
        filename = os.path.join(self.savedir, '{}{}.ini'.format(
            name, self._conf))
        config = configparser.ConfigParser()
        config['general'] = {}
        config['general']['num_data'] = str(dataset.num_data)
        config['general']['metafields'] = str(dataset.metadata_fields)
        config['general']['dtype'] = str(dataset.dtype)
        n_f, f_h, f_w = dataset.accepted_packet_shape
        config['packet_shape'] = {}
        config['packet_shape']['num_frames'] = str(n_f)
        config['packet_shape']['frame_height'] = str(f_h)
        config['packet_shape']['frame_width'] = str(f_w)
        item_types = dataset.item_types
        config['item_types'] = {}
        for k in cons.ALL_ITEM_TYPES:
            config['item_types'][k] = str(item_types[k])
        with open(filename, 'w', encoding='UTF-8') as configfile:
            config.write(configfile)
