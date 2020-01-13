import abc
import os


class FsPersistencyHandler(abc.ABC):

    def __init__(self, load_dir=None, save_dir=None):
        super(FsPersistencyHandler, self).__init__()
        self.loaddir = load_dir
        self.savedir = save_dir

    # properties

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, value):
        if value is not None and not os.path.isdir(value):
            raise IOError('Invalid save directory: {}'.format(value))
        self._savedir = value

    @property
    def loaddir(self):
        return self._loaddir

    @loaddir.setter
    def loaddir(self, value):
        if value is not None and not os.path.isdir(value):
            raise IOError('Invalid load directory: {}'.format(value))
        self._loaddir = value

    def _check_before_write(self, err_msg='Save directory not set'):
        if self.savedir is None:
            raise Exception(err_msg)
        else:
            return True

    def _check_before_read(self, err_msg='Load directory not set'):
        if self.loaddir is None:
            raise Exception(err_msg)
        else:
            return True
