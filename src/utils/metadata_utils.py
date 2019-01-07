import numpy as np


SYNTH_METADATA      = ['bg_lambda', 'num_bad_ECs', 'shower', 'yx_angle', 'max',
                       'duration', 'length', 'start_gtu', 'start_y', 'start_x']
COMMON_METADATA     = ['source_file_acquisition_full', 'start_gtu', 'end_gtu',
                       'packet_id']
FLIGHT_METADATA     = COMMON_METADATA
SIMU_METADATA       = COMMON_METADATA
CLASS_METADATA      = ['item_idx', 'output', 'target', 'shower_prob',
                       'noise_prob']


def extract_metafields(metadata):
    metafields = set()
    for item in metadata:
        metafields = metafields.union(item.keys())
    return metafields


def classification_metadata_handler(raw_output, target, item_idx,
                                    old_dict=None):
    if old_dict == None:
        old_dict = {}
    rnd_output = np.round(raw_output).astype(np.uint8)
    old_dict['shower_prob'] = round(raw_output[0], 6)
    old_dict['noise_prob']  = round(raw_output[1], 6)
    old_dict['output'] = ('shower' if np.array_equal(rnd_output, [1, 0])
                          else 'noise')
    old_dict['target'] = ('shower' if np.array_equal(target, [1, 0])
                          else 'noise')
    old_dict['item_idx'] = item_idx
    return old_dict
