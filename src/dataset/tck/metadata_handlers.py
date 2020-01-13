import dataset.tck.constants as c


class MetadataCreator:

    MANDATORY_PACKET_ATTRS = ('packet_id', 'start_gtu', 'end_gtu')
    MANDATORY_EVENT_META = (c.SRCFILE_KEY, )

    def __init__(self, extra_fields=None):
        self._extra = set(extra_fields or [])

    @property
    def extra_metafields(self):
        return self._extra

    def process_events(self, events):
        """
        Add metadata to events created by the event transformer classes.

        Each element in the iterable is assumed to be a list or other iterable,
        as a single event transformer input entry can be processed into many
        packets (e.g. when using AllPacketsEventTransformer).

        The processed events are returned as a lazily evaluated generator
        expression, which upon iteration returns lists having the same length
        as the corresponding input entry and with each element being a tuple
        with the following structure:

        (packet, processed_metadata_for_packet)

        :param events: events to process
        :type events: iterable of dict
        :returns: generator of tuple
        """
        metafields = self._extra.union(self.MANDATORY_EVENT_META)
        for events_list in events:
            yield [(
                # packet extracted in the previous step
                event['packet'],
                # metadata to include in created dataset
                {
                    "packet_id": event["packet_id"],
                    "start_gtu": event["start_gtu"],
                    "end_gtu": event["end_gtu"],
                    **{fieldname: event['event_meta'][fieldname]
                       for fieldname in metafields}
                }
            ) for event in events_list]
