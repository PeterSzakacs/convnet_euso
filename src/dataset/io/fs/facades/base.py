import abc
import os


class FilesystemPersistenceFacade(abc.ABC):

    @abc.abstractmethod
    def load(self, filename, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, filename, array_like_data, **kwargs):
        pass

    @abc.abstractmethod
    def append(self, filename, array_like_data, **kwargs):
        pass

    def delete(self, filename, **kwargs):
        os.remove(filename)
