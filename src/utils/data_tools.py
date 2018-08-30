import numpy as np

from . import event_reading as reading

def flight_data_to_dataset(acqfile, triggerfile = None, frames_per_packet=128, 
                           start_index_per_packet=27, stop_index_per_packet=47, 
                           frame_width=48, frame_height=48):
    if stop_index_per_packet <= start_index_per_packet:
        raise ValueError('Not a valid index of start ({}) and end ({}) frame within packet'
                            .format(start_index_per_packet, stop_index_per_packet))
    num_frames_to_pick = stop_index_per_packet - start_index_per_packet
    ndarray = np.empty((num_frames_to_pick, frame_width, frame_height), dtype=np.uint8)
    reader = reading.AcqL1EventReader(acqfile, triggerfile)
    iterator = reader.iter_gtu_pdm_data()
    total_num_packets = reader.tevent_entries

    test_idx = 0
    frames_list, targets_list = [], []
    frame_start_in_packet, frame_stop_in_packet = start_index_per_packet, stop_index_per_packet
    ndarray_idx, curr_packet_idx = 0, 0
    for frame in iterator:
        test_idx += 1
        # prevents more than 1 condition being checked on most iterations
        if frame.gtu < frame_start_in_packet:
            continue
        elif frame.gtu == frame_stop_in_packet:
            ndarray_idx = 0
            curr_packet_idx += 1
            frame_start_in_packet = curr_packet_idx * frames_per_packet + start_index_per_packet
            frame_stop_in_packet = curr_packet_idx * frames_per_packet + stop_index_per_packet
            frames_list.append(np.max(ndarray, axis=0))
            targets_list.append([0, 1])
        elif frame.gtu >= frame_start_in_packet:
            ndarray[ndarray_idx] = frame.photon_count_data.astype(np.uint8)
            ndarray_idx += 1
    return frames_list, targets_list

def simu_data_to_dataset(npyfile, triggerfile = None, frames_per_packet=128, 
                         start_index_per_packet=27, stop_index_per_packet=47, 
                         frame_width=48, frame_height=48):
    if stop_index_per_packet <= start_index_per_packet:
        raise ValueError('Not a valid index of start ({}) and end ({}) frame within packet'
                            .format(start_index_per_packet, stop_index_per_packet))
    num_frames_to_pick = stop_index_per_packet - start_index_per_packet
    start_frame = frames_per_packet + start_index_per_packet
    stop_frame = frames_per_packet + stop_index_per_packet
    ndarray = np.load(npyfile)

    orig_max_x_y_arr = np.max(ndarray[start_frame:stop_frame], axis=0)
    noise_frame = np.max(ndarray[0:num_frames_to_pick], axis=0)

    frames = (orig_max_x_y_arr, noise_frame)
    targets = ([1, 0], [0, 1])
    return frames, targets
