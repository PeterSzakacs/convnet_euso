import numpy as np

CLASSIFICATION_TARGETS = {
    'shower': [1, 0],
    'noise': [0, 1],
}

def get_target_name(target_value):
    return [k for k, v in CLASSIFICATION_TARGETS.items()
            if np.array_equal(v, target_value)][0]

def get_target_probabilities(raw_output, precision=4):
    probs = {}
    probs['shower'] = round(raw_output[0], precision)
    probs['noise'] = round(raw_output[1], precision)
    return probs