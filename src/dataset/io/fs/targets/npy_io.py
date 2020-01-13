import os

import numpy as np

import dataset.io.fs.base as fs_io_base


class NumpyTargetsPersistencyHandler(fs_io_base.FsPersistencyHandler):

    # static attributes and methods

    DEFAULT_CLASSIFICATION_TARGETS_FILE_SUFFIX = '_class_targets'

    def __init__(self, load_dir=None, save_dir=None,
                 classification_targets_file_suffix=None):
        super(self.__class__, self).__init__(load_dir, save_dir)
        self._targ = (classification_targets_file_suffix or
                      self.DEFAULT_CLASSIFICATION_TARGETS_FILE_SUFFIX)

    def load_targets(self, name):
        """
            Load dataset targets from secondary storage as a numpy.ndarray.

            Parameters
            ----------
            :param name:        the dataset name/targets filename prefix.
            :type name:         str
        """
        filename = '{}{}.npy'.format(name, self._targ)
        return np.load(os.path.join(self.loaddir, filename))

    def save_targets(self, name, targets):
        """
            Persist the dataset targets into secondary storage as an npy file
            stored in outdir.

            Parameters
            ----------
            :param name:        the dataset name.
            :type name:         str
            :param targets:     targets to save/persist.
            :type targets:      typing.Sequence[numpy.ndarray]
        """
        # save targets
        filename = os.path.join(self.savedir, '{}{}.npy'.format(
            name, self._targ))
        np.save(filename, targets)
        return filename
