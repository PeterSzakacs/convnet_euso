import os

import dataset.io.fs.base as fs_io_base
import dataset.metadata_utils as meta
import utils.io_utils as io_utils


class TSVMetadataPersistencyHandler(fs_io_base.FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_METADATA_FILE_SUFFIX = '_meta'

    def __init__(self, load_dir=None, save_dir=None, metafile_suffix=None):
        super(self.__class__, self).__init__(load_dir, save_dir)
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
