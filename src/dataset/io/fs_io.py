import dataset.data.constants as cons
import dataset.dataset_utils as ds
import dataset.io.fs.config.ini.base as ini_base
import dataset.io.fs.config.ini_io as ini_io
import dataset.io.fs.data as data_io
import dataset.io.fs.meta as meta_io
import dataset.io.fs.targets as targets_io


class DatasetFsPersistencyHandler:

    def __init__(self, load_dir=None, save_dir=None, configfile_suffix=None):
        self._conf = ini_io.IniConfigPersistencyHandler(
            file_suffix=configfile_suffix)
        self.loaddir = load_dir
        self.savedir = save_dir

    # properties

    @property
    def loaddir(self):
        return self._conf.loaddir

    @loaddir.setter
    def loaddir(self, value):
        self._conf.loaddir = value

    @property
    def savedir(self):
        return self._conf.savedir

    @savedir.setter
    def savedir(self, value):
        self._conf.savedir = value

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
        (d_handler, t_handler, m_handler) = self._get_handlers(config)
        data = d_handler.load_data(name, item_types)
        targets = t_handler.load_targets(name, config['targets']['types'])
        metadata = m_handler.load_metadata(
            name, metafields=config['metadata']['fields'])
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
        name, config = dataset.name, dataset.attributes
        (d_handler, t_handler, m_handler) = self._get_handlers(config)

        metadata, metafields = dataset.get_metadata(), dataset.metadata_fields
        m_handler.save_metadata(name, metadata, metafields=metafields,
                                metafields_order=metafields_order)

        targets = {'softmax_class_value': dataset.get_targets()}
        t_handler.save_targets(name, targets)

        data = dataset.get_data_as_dict()
        d_handler.save_data(name, data, dtype=dataset.dtype)

        self._conf.save_config(name, config)

    # private helper methods

    def _get_handlers(self, config):
        loaddir, savedir = self.loaddir, self.savedir
        backend = config['data']['backend']
        data_handler = data_io.HANDLERS[backend](load_dir=loaddir,
                                                 save_dir=savedir)
        backend = config['targets']['backend']
        targets_handler = targets_io.HANDLERS[backend](load_dir=loaddir,
                                                       save_dir=savedir)
        backend = config['metadata']['backend']
        metadata_handler = meta_io.HANDLERS[backend](load_dir=loaddir,
                                                     save_dir=savedir)
        return data_handler, targets_handler, metadata_handler
