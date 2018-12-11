import numpy as np

import utils.data_templates as templates
import libs.event_reading as reading


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
                              ' evenly divisible intopackets of size {} frames'
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

        num_packets = frames_total/self._template.num_frames
        container_shape = (num_packets, *self._template.packet_shape)
        dtype = first_frame.dtype 
        packets = np.empty(container_shape, dtype=dtype)
        iterator = reader.iter_gtu_pdm_data()
        frame_idx, packet_idx = 0, 0
        for frame in iterator:
            packet_idx = frame_idx / self._template.num_frames
            packets[packet_idx, frame_idx] = frame.photon_count_data
            frame_idx += 1
        return packets

    def extract_packets_from_npyfile(self, npyfile, triggerfile=None):
        ndarray = np.load(npyfile)
        frame_shape  = ndarray.shape[1:]
        frames_total = len(ndarray)

        self._check_packet_against_template(frame_shape, frames_total, npyfile)

        num_packets = int(frames_total/self._template.num_frames)
        return ndarray.reshape(num_packets, *self._template.packet_shape)
