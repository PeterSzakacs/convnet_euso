import os
import typing as t

import dataset.constants as cons
import dataset.dataset_utils as ds
import dataset.io.fs.base as fs_io_base
import dataset.io.fs.config.ini_io as ini_io
import dataset.io.fs.data.managers as data_io
import dataset.io.fs.meta.tsv_io as meta_io
import dataset.io.fs.targets.managers as targets_io


class DatasetFsPersistencyHandler(fs_io_base.FsPersistencyHandler):

    def __init__(
            self,
            load_dir=None,
            save_dir=None,
            data_handler=None,
            targets_handler=None,
            metadata_handler=None
    ):
        super(self.__class__, self).__init__(load_dir, save_dir)
        self._conf = ini_io.IniConfigPersistenceManager()
        self._data_handler = (data_handler or
                              data_io.FilesystemDataManager())
        self._target_handler = (targets_handler or
                                targets_io.FilesystemTargetsManager())
        self._meta_handler = (metadata_handler or
                              meta_io.TSVMetadataPersistencyHandler())

    # dataset load

    def load_dataset_config(
            self,
            name: str
    ):
        """
            Loads the configuration of a dataset from secondary storage.

            Essentially, this function creates a dictionary of all 'public'
            attributes and properties of an existing dataset without loading
            its data, targets and metadata into memory.

            Parameters
            ----------
            :param name:        the dataset name/config filename prefix.
        """
        self._check_before_read()
        filename = '{}_{}'.format(name, 'config.ini')
        filename = os.path.join(self.loaddir, filename)

        config = self._conf.load(filename)
        config['data']['num_items'] = config['num_items']
        config['targets']['num_items'] = config['num_items']
        return config

    def load_empty_dataset(
            self,
            name: str,
            item_types: t.Mapping[str, bool] = None
    ):
        """
            Create a dataset from configuration stored in secondary storage
            without loading any of its actual contents (data, targets,
            metadata).

            Parameters
            ----------
            :param name:        the dataset name/config filename prefix.
            :param item_types:  (optional) types of dataset items to load.
        """
        config = self.load_dataset_config(name)
        load_types = self._resolve_load_types(config, item_types)
        return self._to_dataset(name, config, load_types)

    def load_dataset(
            self,
            name: str,
            item_types: t.Mapping[str, bool] = None
    ):
        """
            Load a dataset from secondary storage.

            This function assumes that the relevant dataset files are located
            in the same directory (loaddir).

            Parameters
            ----------
            :param name:        the dataset name.
            :param item_types:  (optional) types of dataset items to load.
        """
        config = self.load_dataset_config(name)
        load_types = self._resolve_load_types(config, item_types)
        dataset = self._to_dataset(name, config, load_types)

        data = self._data_handler.load(name, self.loaddir,
                                       config['data'], load_types=load_types)
        targets = self._target_handler.load(name, self.loaddir,
                                            config['targets'])
        self._meta_handler.loaddir = self.loaddir
        metadata = self._meta_handler.load_metadata(name)

        # TODO: Think of a way to load dataset with items that does not depend
        # on knowledge of NumpyDataset internals
        # currently excluded from unit tests for that very reason
        dataset._data.extend(data)
        dataset._targ.extend(
            {'classification': targets['softmax_class_value']}
        )
        dataset._meta.extend(metadata)
        dataset._num_data = config['num_items']
        return dataset

    # dataset save/persist

    def save_dataset(
            self,
            dataset: ds.NumpyDataset,
            metafields_order: t.Sequence[str] = None
    ):
        """
            Persist the dataset into secondary storage, with all files stored
            in the same directory (outdir).

            Parameters
            ----------
            :param dataset:        the dataset to save/persist.
            :param metafields_order:    (optional) ordering of fields (columns)
                                        in the created metadata TSV.
        """
        self._check_before_write()
        name, config = dataset.name, dataset.attributes
        filename = '{}_{}'.format(name, 'config.ini')
        filename = os.path.join(self.savedir, filename)
        self._conf.save(filename, config)

        metadata, metafields = dataset.get_metadata(), dataset.metadata_fields
        self._meta_handler.savedir = self.savedir
        self._meta_handler.save_metadata(name, metadata, metafields=metafields,
                                         metafields_order=metafields_order)

        targets = {'softmax_class_value': dataset.get_targets()}
        self._target_handler.save(name, self.savedir, config['targets'],
                                  targets)

        data = dataset.get_data_as_dict()
        self._data_handler.save(name, self.savedir, config['data'], data)

    @staticmethod
    def _resolve_load_types(config, item_types):
        if not item_types:
            return set(config['data']['types'])
        else:
            return set(it for it, is_present in item_types.items()
                       if is_present)

    @staticmethod
    def _to_dataset(name, config, load_types):
        dtype = next(iter(config['data']['types'].values()))['dtype']
        itypes = dict.fromkeys(cons.ALL_ITEM_TYPES, False)
        itypes.update({k: True for k in load_types})
        dataset = ds.NumpyDataset(name, config['data']['packet_shape'],
                                  item_types=itypes, dtype=dtype)
        return dataset
