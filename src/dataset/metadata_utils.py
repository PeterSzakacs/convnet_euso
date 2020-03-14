import collections


def extract_metafields(metadata):
    metafields = set()
    for item in metadata:
        metafields = metafields.union(item.keys())
    return metafields


class MetadataHolder:

    def __init__(self, **attrs):
        self._metadata = []
        self._metafields = set()
        self._attrs = attrs

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, idx):
        if idx is None:
            return self._metadata
        elif isinstance(idx, (slice, int)):
            return self._metadata[idx]
        elif isinstance(idx, collections.Sequence):
            # range, list, tuple, etc
            return [self._metadata[index] for index in idx]
        else:
            raise Exception('Unsupported index type: {}'.format(type(idx)))

    @property
    def attributes(self):
        return {
            'backend': 'tsv',
            **self._attrs,
            'fields': self.metadata_fields,
        }

    @property
    def metadata_fields(self):
        """
            Fields of dataset item-level metadata.
        """
        return self._metafields

    def append(self, meta_dict):
        """
            Append new metadata item to the holder
        """
        self._metafields = self._metafields.union(meta_dict.keys())
        self._metadata.append(meta_dict)

    def extend(self, meta_dict_iterable):
        """
            Append batch of new metadata to the holder from an iterable

            Parameters
            ----------
            :param meta_dict_iterable:   iterable of metadata
            :type meta_dict_iterable:    typing.Iterable[typing.Mapping[
                                            str, any]]
        """
        self._metadata.extend(meta_dict_iterable)
        self._metafields = extract_metafields(self._metadata)

    def add_metafield(self, name, default_value=None):
        """
            Add a new metafield with optional default value to the holder.

            Parameters
            ----------
            :param name:   name of the field
            :type name:    str
            :param default_value:  default value for the new metafield
            :type default_value:   any
        """
        self._metafields = self._metafields.union([name])
        for metadata in self._metadata:
            metadata[name] = default_value

    def shuffle(self, shuffler):
        """
            Shuffle the contained metadata using the provided shuffler
        """
        shuffler(self._metadata)
