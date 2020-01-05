import dataset.tck.constants as c


class MetadataCreator:

    MANDATORY_PACKET_ATTRS = ('packet_id', 'start_gtu', 'end_gtu')
    MANDATORY_EVENT_META = (c.SRCFILE_KEY, )

    def __init__(self, extra_fields=None):
        self._extra = set(extra_fields or [])

    @property
    def extra_metafields(self):
        return self._extra

    def create_metadata(self, packet_attrs, event_metadata):
        metadata = []
        metafields = self._extra.union(self.MANDATORY_EVENT_META)
        meta_dict = {field: event_metadata.get(field) for field in metafields}
        for packet_attr in packet_attrs:
            meta = meta_dict.copy()
            for fieldname in self.MANDATORY_PACKET_ATTRS:
                meta[fieldname] = packet_attr[fieldname]
            metadata.append(meta)
        return metadata
