LAYER_CATEGORIES = ('input', 'hidden', 'trainable')

LAYER_TYPES = ('Conv2D', 'LRN', 'MaxPool2D', 'FC', 'Dropout', 'Merge',
               'Flatten', 'Reshape', 'Input', 'Upsample2D')

# well-defined modes to use when splitting dataset for training
DATASET_SPLIT_MODES = ('FROM_START', 'FROM_END', 'RANDOM', )

# dict keys to identify data and target subsets for training and testing
TRAIN_DATA_DICT_KEYS = ('train_data', 'train_targets', 'test_data',
                        'test_targets', )