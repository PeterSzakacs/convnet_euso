# module level constants

# data

ALL_ITEM_TYPES = ('raw', 'yx', 'gtux', 'gtuy')

# targets

TARGET_TYPES = ['classification']

CLASSIFICATION_TARGETS = {
    'shower': [1, 0],
    'noise': [0, 1],
}

# metadata

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
