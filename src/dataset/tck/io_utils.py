import collections

import numpy as np

import libs.event_reading as reading
import utils.data_templates as templates

# classes


class PacketExtractor:

    def __init__(self, packet_template=templates.PacketTemplate(
                       16, 16, 48, 48, 128)):
        self.packet_template = packet_template

    @property
    def packet_template(self):
        return self._template

    @packet_template.setter
    def packet_template(self, value):
        if value is None or not isinstance(value, templates.PacketTemplate):
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


class PacketCache:

    def __init__(self, max_size, packet_extractors, num_evict_on_full=10):
        if max_size < num_evict_on_full:
            raise ValueError('Number of evicted items must be less than the '
                             'cache size')
        self._maxsize = max_size
        self._extractors = {}
        for key in ('NPY', 'ROOT'):
            self._extractors[key] = packet_extractors[key]
        self._num_evict = num_evict_on_full
        self._packets = {}
        self._file_queue = collections.deque([], max_size)

    def get(self, filename):
        all_packets, queue = self._packets, self._file_queue
        packets = all_packets.get(filename, None)
        if packets is None:
            extractors = self._extractors
            if filename.endswith('.npy'):
                extractor = extractors['NPY']
            elif filename.endswith('.root'):
                extractor = extractors['ROOT']
            else:
                raise Exception('Unknown file type: {}'.format(filename))
            packets = extractor(filename)
            all_packets[filename] = packets
            queue.append(filename)
            if len(queue) == self._maxsize:
                for idx in range(self._num_evict):
                    filename = queue.popleft()
                    all_packets.pop(filename)
        return packets
