import dataset.data.constants as cons
import dataset.dataset_utils as ds
import dataset.io.fs.config.ini.base as ini_base
import dataset.io.fs.config.ini_io as ini_io
import dataset.io.fs.data.npy_io as data_io
import dataset.io.fs.meta.tsv_io as meta_io
import dataset.io.fs.targets.npy_io as targets_io


class DatasetFsPersistencyHandler:

    def __init__(self, load_dir=None, save_dir=None, configfile_suffix=None,
                 data_handler=None, targets_handler=None,
                 metadata_handler=None):
        self._conf = ini_io.IniConfigPersistencyHandler(
            file_suffix=configfile_suffix)
        self._data_handler = (data_handler or
                              data_io.NumpyDataPersistencyHandler())
        self._target_handler = (targets_handler or
                                targets_io.NumpyTargetsPersistencyHandler())
        self._meta_handler = (metadata_handler or
                              meta_io.TSVMetadataPersistencyHandler())
        self.loaddir = load_dir
        self.savedir = save_dir

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

    # property overrides

    @property
    def loaddir(self):
        return self._conf.loaddir

    @loaddir.setter
    def loaddir(self, value):
        self._conf.loaddir = value
        self._data_handler.loaddir = value
        self._target_handler.loaddir = value
        self._meta_handler.loaddir = value

    @property
    def savedir(self):
        return self._conf.savedir

    @savedir.setter
    def savedir(self, value):
        self._conf.savedir = value
        self._data_handler.savedir = value
        self._target_handler.savedir = value
        self._meta_handler.savedir = value

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
        config_handler = self._conf
        version = config_handler.get_config_version(name)
        if version != config_handler.config_parser.version:
            config_handler.config_parser = ini_base.get_ini_parser(version)
        return config_handler.load_config(name)

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
        dtype = next(iter(attrs['data']['types'].values()))['dtype']
        item_types = item_types or dict.fromkeys(attrs['data']['types'], True)
        itypes = dict.fromkeys(cons.ALL_ITEM_TYPES, False)
        itypes.update(item_types)
        dataset = ds.NumpyDataset(name, attrs['data']['packet_shape'],
                                  item_types=itypes, dtype=dtype)
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
        config = self.load_dataset_config(name)
        item_types = item_types or dict.fromkeys(config['data']['types'], True)
        itypes = dict.fromkeys(cons.ALL_ITEM_TYPES, False)
        itypes.update(item_types)
        dataset = ds.NumpyDataset(name, config['data']['packet_shape'],
                                  item_types=itypes)
        item_types = {itype for itype, is_present in dataset.item_types.items()
                      if is_present}
        data = self._data_handler.load_data(name, item_types)
        targets = self._target_handler.load_targets(name)
        metadata = self._meta_handler.load_metadata(name)
        dataset._data.extend(data)
        dataset._targ.extend({'classification': targets})
        dataset._meta.extend(metadata)
        dataset._num_data = config['num_items']
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
        name = dataset.name
        metadata, metafields = dataset.get_metadata(), dataset.metadata_fields
        self._meta_handler.save_metadata(name, metadata, metafields=metafields,
                                         metafields_order=metafields_order)
        targets = dataset.get_targets()
        self._target_handler.save_targets(name, targets)
        data = dataset.get_data_as_dict()
        self._data_handler.save_data(name, data, dtype=dataset.dtype)

        self._conf.save_config(name, dataset.attributes)
