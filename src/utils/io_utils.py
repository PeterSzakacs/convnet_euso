import abc
import ast
import configparser
import os
import csv

import numpy as np

import utils.dataset_utils as ds
import utils.data_templates as templates
import utils.metadata_utils as meta
import libs.event_reading as reading


def load_TSV(filename, selected_columns=None, output_list=None):
    output_list = output_list or []
    if selected_columns is not None:
        process_row = lambda row, cols: {col:row[col] for col in cols}
    else:
        process_row = lambda row, cols: row
    with open(filename, 'r', encoding='UTF-8') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in reader:
            output_list.append(process_row(row, selected_columns))
    return output_list


def save_TSV(filename, rows, column_order, file_exists_overwrite=False):
    if os.path.isfile(filename) and not file_exists_overwrite:
        raise FileExistsError('Cannot overwrite existing file'
                              .format(filename))
    with open(filename, 'w', encoding='UTF-8') as outfile:
        writer = csv.DictWriter(outfile, column_order, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


class packet_extractor():

    def __init__(self, packet_template=templates.packet_template(
                       16, 16, 48, 48, 128)):
        self.packet_template = packet_template

    @property
    def packet_template(self):
        return self._template

    @packet_template.setter
    def packet_template(self, value):
        if (value is None or not isinstance(value, templates.packet_template)):
            raise TypeError("Not a valid packet template object: {}".format(
                            value))
        self._template = value

    def _check_packet_against_template(self, frame_shape, total_num_frames,
                                       srcfile):
        frames_per_packet = self._template.num_frames
        if total_num_frames % frames_per_packet != 0:
            raise ValueError(('The total number of frames ({}) in {} is not'
                              ' evenly divisible to packets of size {} frames'
                              ).format(total_num_frames, srcfile,
                                       frames_per_packet))
        exp_frame_shape = self._template.packet_shape[1:]
        if frame_shape != exp_frame_shape:
            raise ValueError(('The width or height of frames ({}) in {} does'
                              ' not match that of the template ({})').format(
                              frame_shape, srcfile, exp_frame_shape))

    def extract_packets_from_rootfile(self, acqfile, triggerfile=None):
        reader = reading.AcqL1EventReader(acqfile, triggerfile)
        iterator = reader.iter_gtu_pdm_data()
        first_frame = next(iterator).photon_count_data
        # NOTE: ROOT file iterator returns packet frames of shape
        # (1, 1, height, width)
        frame_shape = first_frame.shape[2:4]
        frames_total = reader.tevent_entries

        self._check_packet_against_template(frame_shape, frames_total, acqfile)

        num_frames = self._template.num_frames
        num_packets = int(frames_total / num_frames)
        container_shape = (num_packets, *self._template.packet_shape)
        dtype = first_frame.dtype
        packets = np.empty(container_shape, dtype=dtype)
        # reset iterator to start of packets list
        iterator = reader.iter_gtu_pdm_data()
        for frame in iterator:
            global_gtu = frame.gtu
            packet_idx = int(global_gtu / num_frames)
            packet_gtu = global_gtu % num_frames
            packets[packet_idx][packet_gtu] = frame.photon_count_data
        return packets

    def extract_packets_from_npyfile(self, npyfile, triggerfile=None):
        ndarray = np.load(npyfile)
        frame_shape  = ndarray.shape[1:]
        frames_total = len(ndarray)

        self._check_packet_against_template(frame_shape, frames_total, npyfile)

        num_packets = int(frames_total / self._template.num_frames)
        return ndarray.reshape(num_packets, *self._template.packet_shape)


class fs_persistency_handler(abc.ABC):

    def __init__(self, load_dir=None, save_dir=None):
        super(fs_persistency_handler, self).__init__()
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


class dataset_metadata_fs_persistency_handler(fs_persistency_handler):

    # static attributes and methods

    DEFAULT_METADATA_FILE_SUFFIX = '_meta'

    def __init__(self, load_dir=None, save_dir=None, metafile_suffix=None):
        super(dataset_metadata_fs_persistency_handler, self).__init__(
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
        meta = load_TSV(filename, selected_columns=meta_fields)
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
        save_TSV(filename, metadata, metafields_order,
                 file_exists_overwrite=True)
        return filename


class dataset_targets_fs_persistency_handler(fs_persistency_handler):

    # static attributes and methods

    DEFAULT_CLASSIFICATION_TARGETS_FILE_SUFFIX = '_class_targets'

    def __init__(self, load_dir=None, save_dir=None,
                 classification_targets_file_suffix=None):
        super(dataset_targets_fs_persistency_handler, self).__init__(
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


class dataset_fs_persistency_handler(fs_persistency_handler):

    # static attributes and methods

    DEFAULT_CONFIG_FILE_SUFFIX = '_config'
    DEFAULT_DATA_FILES_SUFFIXES = {k: '_{}'.format(k)
                                   for k in ds.ALL_ITEM_TYPES}

    def __init__(self, load_dir=None, save_dir=None, data_files_suffixes={},
                 configfile_suffix=None, targets_handler=None,
                 metadata_handler=None):
        super(dataset_fs_persistency_handler, self).__init__(
            load_dir, save_dir)
        self._conf = configfile_suffix or self.DEFAULT_CONFIG_FILE_SUFFIX
        self._data = {}
        for k in ds.ALL_ITEM_TYPES:
            self._data[k] = data_files_suffixes.get(
                k, self.DEFAULT_DATA_FILES_SUFFIXES[k])
        self._target_handler = (targets_handler or
                                dataset_targets_fs_persistency_handler(
                                    load_dir=load_dir, save_dir=save_dir))
        self._meta_handler   = (metadata_handler or
                                dataset_metadata_fs_persistency_handler(
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
        dataset = ds.numpy_dataset(name, attrs['packet_shape'], itypes)
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
        ds.check_item_types(item_types)
        data = {}
        for item_type in ds.ALL_ITEM_TYPES:
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
        # on knowledge of numpy_dataset internals
        # currently excluded from unit tests for that very reason
        self._check_before_read()
        config = self.load_dataset_config(name)
        itypes = item_types or config['item_types']
        dataset = ds.numpy_dataset(name, config['packet_shape'], itypes)
        data = self.load_data(name, dataset.item_types)
        targets = self._target_handler.load_targets(name)
        metadata = self._meta_handler.load_metadata(name)
        for itype, is_present in dataset.item_types.items():
            if is_present:
                dataset._data[itype].extend(data[itype])
        dataset._targets.extend(targets)
        dataset._metadata.extend(metadata)
        dataset._metafields = config['metafields']
        dataset._num_data = config['num_data']
        return dataset

    # dataset save/persist

    def save_data(self, name, data_items_dict):
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
        """
        self._check_before_write()
        savefiles = {}
        # save data
        keys = set(ds.ALL_ITEM_TYPES).intersection(set(data_items_dict.keys()))
        for k in keys:
            filename = os.path.join(self.savedir, '{}{}.npy'.format(
                name, self._data[k]))
            np.save(filename, data_items_dict[k])
            savefiles[k] = filename
        return savefiles

    def save_dataset(self, dataset, metafields_order=None):
        """
            Persist the dataset into secondary storage, with all files stored
            in the same directory (outdir).

            Parameters
            ----------
            :param dataset:        the dataset to save/persist.
            :type dataset:         utils.dataset_utils.numpy_dataset
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
        self.save_data(name, data)

        # save configuration file
        filename = os.path.join(self.savedir, '{}{}.ini'.format(
            name, self._conf))
        config = configparser.ConfigParser()
        config['general'] = {}
        config['general']['num_data'] = str(dataset.num_data)
        config['general']['metafields'] = str(dataset.metadata_fields)
        n_f, f_h, f_w = dataset.accepted_packet_shape
        config['packet_shape'] = {}
        config['packet_shape']['num_frames'] = str(n_f)
        config['packet_shape']['frame_height'] = str(f_h) 
        config['packet_shape']['frame_width'] = str(f_w)
        item_types = dataset.item_types
        config['item_types'] = {}
        for k in ds.ALL_ITEM_TYPES:
            config['item_types'][k] = str(item_types[k])
        with open(filename, 'w', encoding='UTF-8') as configfile:
            config.write(configfile)

