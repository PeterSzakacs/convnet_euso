import numpy as np

SYNTH_METADATA      = ['bg_lambda', 'num_bad_ECs', 'shower', 'yx_angle', 'max',
                       'duration', 'length', 'start_gtu', 'start_y', 'start_x']
# COMMON_METADATA     = ['source_file_acquisition_full', 'start_gtu', 'end_gtu',
#                        'packet_id']
FLIGHT_METADATA     = ['source_file_acquisition_full', 'start_gtu', 'end_gtu',
                       'packet_id']
SIMU_METADATA       = ['source_file_acquisition_full', 'start_gtu', 'end_gtu',
                       'packet_id']


METADATA_TYPES = {
    'flight': {'field order': FLIGHT_METADATA},
    'simu': {'field order': SIMU_METADATA},
    'synth': {'field order': SYNTH_METADATA}
}


def extract_metafields(metadata):
    metafields = set()
    for item in metadata:
        metafields = metafields.union(item.keys())
    return metafields
