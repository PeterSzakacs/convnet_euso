import numpy as np

from .packets import packet_utils as pack
from . import event_reading as reading

class packet_extractor():

    def __init__(self, packet_template=pack.packet_template(16, 16, 48, 48, 128)):
        self.packet_template = packet_template
    
    @property
    def packet_template(self):
        return self._template

    @packet_template.setter
    def packet_template(self, value):
        if (value == None or not isinstance(value, pack.packet_template)):
            raise TypeError("Not a valid packet template object: {}".format(value))
        self._template = value

    def _check_packet_against_template(self, frame_shape, total_num_frames, srcfile):
        num_frames_per_packet = self._template.num_frames
        if total_num_frames % num_frames_per_packet != 0:
            raise ValueError(("Invalid packet template: The total number of frames ({}) in {} "
                                "is not evenly divisible into packets of size {} frames")
                                .format(total_num_frames, srcfile, num_frames_per_packet))
        if (frame_shape != self._template.packet_shape[1:]):
            raise ValueError(("Invalid packet template: The width or height of frames ({}) in {} "
                                "does not match that of the template ({})")
                                .format(frame_shape, srcfile, self._template.packet_shape[1:]))


    def extract_packets_from_rootfile_and_process(self, acqfile, triggerfile = None, 
                                                    on_packet_extracted=lambda packet, packet_idx, srcfile: None):
        reader = reading.AcqL1EventReader(acqfile, triggerfile)
        iterator = reader.iter_gtu_pdm_data()
        first_frame_shape = next(iterator).photon_count_data.shape
        total_num_frames = reader.tevent_entries

        self._check_packet_against_template(first_frame_shape, total_num_frames, acqfile)

        packets = np.empty((total_num_frames, self._template.frame_width, self._template.frame_height))
        iterator = reader.iter_gtu_pdm_data()
        frame_idx, num_frames = 0, self._template.num_frames
        next_packet_idx, curr_packet_idx = 0, 0
        for frame in iterator:
            packets[frame_idx] = frame.photon_count_data
            frame_idx += 1
            curr_packet_idx = next_packet_idx
            next_packet_idx = int(frame_idx / num_frames)
            if next_packet_idx != curr_packet_idx:
                packet_start, packet_stop = curr_packet_idx*num_frames, next_packet_idx*num_frames
                on_packet_extracted(packets[packet_start:packet_stop], curr_packet_idx, acqfile)

    def extract_packets_from_npyfile_and_process(self, npyfile, triggerfile = None, 
                                                    on_packet_extracted=lambda packet, packet_idx, srcfile: None):
        ndarray = np.load(npyfile)
        frame_shape = ndarray.shape[1:]
        total_num_frames = len(ndarray)

        self._check_packet_against_template(frame_shape, total_num_frames, npyfile)

        num_frames = self._template.num_frames
        total_num_packets = int(total_num_frames / num_frames)
        for packet_idx in range(total_num_packets):
            packet_start, packet_stop = packet_idx*num_frames, (packet_idx+1)*num_frames
            packet = ndarray[packet_start:packet_stop]
            on_packet_extracted(packet, packet_idx, npyfile)
