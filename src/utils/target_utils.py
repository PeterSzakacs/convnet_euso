import numpy as np

TARGET_TYPES = ['classification']

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

class TargetsHolder():

    def __init__(self, target_types={'classification': True}):
        self._num_targets = 0
        self._target_types = target_types
        targets = dict.fromkeys(TARGET_TYPES)
        used_types = {}
        for target_type in target_types:
            if target_type:
                targets[target_type] = []
                used_types[target_type] = True
        self._targets = targets
        self._used_types = used_types

    def append(self, targets_dict):
        for ttype in self._used_types.keys():
            self._targets[ttype].append(targets_dict[ttype])

    def extend(self, targets_iterable_dict):
        for ttype in self._used_types.keys():
            self._targets[ttype].extend(targets_iterable_dict[ttype])

    def shuffle(self, shuffler, shuffler_state_resetter):
        for target_type, target in self._targets.items():
            shuffler(target)
            shuffler_state_resetter()

    def get_targets_as_araylike(self, targets_slice_or_idx=None):
        s = (slice(None) if targets_slice_or_idx is None
             else targets_slice_or_idx)
        return tuple(self._targets[ttype][s] for ttype in TARGET_TYPES
                     if self._target_types[ttype])

    def get_targets_as_dict(self, targets_slice_or_idx=None):
        s = (slice(None) if targets_slice_or_idx is None
             else targets_slice_or_idx)
        return {ttype: self._targets[ttype][s] for ttype in TARGET_TYPES
                if self._target_types[ttype]}
