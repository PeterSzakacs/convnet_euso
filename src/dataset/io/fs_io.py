import ast
import configparser
import os

import dataset.constants as cons
import dataset.dataset_utils as ds
import dataset.io.fs.base as fs_io_base
import dataset.io.fs.data.npy_io as data_io
import dataset.io.fs.meta.tsv_io as meta_io
import dataset.io.fs.targets.npy_io as targets_io


class DatasetFsPersistencyHandler(fs_io_base.FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_CONFIG_FILE_SUFFIX = '_config'

    def __init__(self, load_dir=None, save_dir=None, configfile_suffix=None,
                 data_handler=None, targets_handler=None,
                 metadata_handler=None):
        super(self.__class__, self).__init__(load_dir, save_dir)
        self._conf = configfile_suffix or self.DEFAULT_CONFIG_FILE_SUFFIX
        self._data_handler   = (data_handler or
                                data_io.NumpyDataPersistencyHandler(
                                    load_dir=load_dir, save_dir=save_dir))
        self._target_handler = (targets_handler or
                                targets_io.NumpyTargetsPersistencyHandler(
                                    load_dir=load_dir, save_dir=save_dir))
        self._meta_handler   = (metadata_handler or
                                meta_io.TSVMetadataPersistencyHandler(
                                    load_dir=load_dir, save_dir=save_dir))

    # properties

    @property
    def data_persistency_handler(self):
        return self._data_handler

    @property
    def targets_persistency_handler(self):
        return self._target_handler

    @property
    def metadata_persistency_handler(self):
        return self._meta_handler

    # dataset load

    def load_dataset_config(self, name):
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
        data = self._data_handler.load_data(name, dataset.item_types)
        targets = self._target_handler.load_targets(name)
        metadata = self._meta_handler.load_metadata(name)
        dataset._data.extend(data)
        dataset._targ.extend({'classification': targets})
        dataset._meta.extend(metadata)
        dataset._num_data = config['num_data']
        return dataset

    # dataset save/persist

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
        self._data_handler.save_data(name, data, dtype=dataset.dtype)

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
