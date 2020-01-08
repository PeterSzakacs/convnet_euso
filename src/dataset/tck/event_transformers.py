import dataset.tck.constants as c


class DefaultEventTransformer:

    REQUIRED_FILELIST_COLUMNS = (c.SRCFILE_KEY, 'packet_id', )

    def __init__(self, packets_extraction_fn, packet_id,
                 start_gtu, stop_gtu):
        self._extraction_fn = packets_extraction_fn
        self._packet_id = packet_id
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def process_events(self, events):
        packets_extraction_fn = self._extraction_fn
        idx, start, stop = self._packet_id, self._start_gtu, self._stop_gtu
        for event in events:
            srcfile = event[c.SRCFILE_KEY]
            packets = packets_extraction_fn(srcfile)
            result = {'packet': packets[idx][start:stop], 'packet_id': idx,
                      'start_gtu': start, 'end_gtu': stop, 'event_meta': event}
            yield [result, ]


class AllPacketsEventTransformer:

    REQUIRED_FILELIST_COLUMNS = (c.SRCFILE_KEY, )

    def __init__(self, packets_extraction_fn, start_gtu, stop_gtu):
        self._extraction_fn = packets_extraction_fn
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def process_events(self, events):
        packets_extraction_fn = self._extraction_fn
        start, stop = self._start_gtu, self._stop_gtu
        for event in events:
            srcfile = event[c.SRCFILE_KEY]
            packets = packets_extraction_fn(srcfile)
            yield [{'packet': packets[idx][start:stop], 'packet_id': idx,
                   'start_gtu': start, 'end_gtu': stop, 'event_meta': event}
                   for idx in range(len(packets))]


class GtuInPacketEventTransformer:

    REQUIRED_FILELIST_COLUMNS = (c.SRCFILE_KEY, 'packet_id', 'gtu_in_packet')

    def __init__(self, packets_extraction_fn, adjust_if_out_of_bounds=True,
                 num_gtu_before=None, num_gtu_after=None):
        self._extraction_fn = packets_extraction_fn
        self._gtu_before = num_gtu_before or 4
        self._gtu_after = num_gtu_after or 15
        self._gtu_after = self._gtu_after + 1
        self._adjust = adjust_if_out_of_bounds

    @property
    def num_frames(self):
        return self._gtu_after + self._gtu_before

    def process_events(self, events):
        packets_extraction_fn = self._extraction_fn
        for event in events:
            idx, gtu = int(event['packet_id']), int(event['gtu_in_packet'])
            srcfile = event[c.SRCFILE_KEY]
            packet = packets_extraction_fn(srcfile)[idx]
            start = gtu - self._gtu_before
            stop = gtu + self._gtu_after
            if (start < 0 or stop > packet.shape[0]) and not self._adjust:
                idx = event.get(['event_id'], srcfile)
                raise Exception('Frame range for event id {} ({}:{}) is out of'
                                ' packet bounds'.format(idx, start, stop))
            else:
                while start < 0:
                    start += 1
                    stop += 1
                while stop > packet.shape[0]:
                    start -= 1
                    stop -= 1
            result = {'packet': packet[start:stop], 'packet_id': idx,
                      'start_gtu': start, 'end_gtu': stop, 'event_meta': event}
            yield [result, ]
