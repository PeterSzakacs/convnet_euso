import os

# default log directorites for tensorboard logs
DEFAULT_CHECK_LOGDIR = '/run/user/{}/convnet_checker'.format(os.getuid())
DEFAULT_TRAIN_LOGDIR = '/run/user/{}/convnet_trainer'.format(os.getuid())
DEFAULT_XVAL_LOGDIR = '/run/user/{}/convnet_xvalidator'.format(os.getuid())

# well-defined modes to use when splitting dataset for training
DATASET_SPLIT_MODES = ('FROM_START', 'FROM_END', 'RANDOM', )

# dict keys to identify data and target subsets for training and testing
TRAIN_DATA_DICT_KEYS = ('train_data', 'train_targets', 'test_data',
                        'test_targets', )