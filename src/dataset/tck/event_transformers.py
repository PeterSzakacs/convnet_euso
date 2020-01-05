import dataset.tck.constants as c


class DefaultEventTransformer:

    REQUIRED_FILELIST_COLUMNS = ('packet_id', )

    def __init__(self, packet_id, start_gtu, stop_gtu):
        self._packet_id = packet_id
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def event_to_packets(self, event_packets, event_metadata):
        idx = self._packet_id
        start, stop = self._start_gtu, self._stop_gtu
        result = {'packet': event_packets[idx][start:stop], 'packet_id': idx,
                  'start_gtu': start, 'end_gtu': stop, }
        return [result, ]


class AllPacketsEventTransformer:

    REQUIRED_FILELIST_COLUMNS = ()

    def __init__(self, start_gtu, stop_gtu):
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def event_to_packets(self, event_packets, event_metadata):
        results = []
        start, stop = self._start_gtu, self._stop_gtu
        for idx in range(len(event_packets)):
            packet = event_packets[idx][start:stop]
            result = {'packet': packet, 'packet_id': idx,
                      'start_gtu': start, 'end_gtu': stop, }
            results.append(result)
        return results


class GtuInPacketEventTransformer:

    REQUIRED_FILELIST_COLUMNS = ('packet_id', 'gtu_in_packet')

    def __init__(self, num_gtu_before=None, num_gtu_after=None,
                 adjust_if_out_of_bounds=True):
        self._gtu_before = num_gtu_before or 4
        self._gtu_after = num_gtu_after or 15
        self._gtu_after = self._gtu_after + 1
        self._adjust = adjust_if_out_of_bounds

    @property
    def num_frames(self):
        return self._gtu_after + self._gtu_before

    def event_to_packets(self, event_packets, event_metadata):
        idx = int(event_metadata['packet_id'])
        packet = event_packets[idx]
        packet_gtu = int(event_metadata['gtu_in_packet'])
        start = packet_gtu - self._gtu_before
        stop = packet_gtu + self._gtu_after
        if (start < 0 or stop > packet.shape[0]) and not self._adjust:
            idx = event_metadata.get(['event_id'],
                                     event_metadata[c.SRCFILE_KEY])
            raise Exception('Frame range for event id {} ({}:{}) is out of '
                            'packet bounds'.format(idx, start, stop))
        else:
            while start < 0:
                start += 1
                stop += 1
            while stop > packet.shape[0]:
                start -= 1
                stop -= 1
        result = {'packet': packet[start:stop], 'packet_id': idx,
                  'start_gtu': start, 'end_gtu': stop}
        return [result, ]
