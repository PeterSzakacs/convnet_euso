import abc
import os
import typing


def append_file_extension(filenames, extension):
    if isinstance(filenames, str):
        return f'{filenames}.{extension}'
    elif isinstance(filenames, typing.Iterable):
        return (f'{name}.{extension}' for name in filenames)
    else:
        raise ValueError(f"Unsupported type passed in for 'filenames' param "
                         f"(expected str or Iterable, not {type(filenames)})")


def create_full_path(filenames, common_dir):
    join = os.path.join
    if isinstance(filenames, str):
        return join(common_dir, filenames)
    elif isinstance(filenames, typing.Iterable):
        return (join(common_dir, name) for name in filenames)
    else:
        raise ValueError(f"Unsupported type passed in for 'filenames' param "
                         f"(expected str or Iterable, not {type(filenames)})")


class FilenameFormatter(abc.ABC):

    @abc.abstractmethod
    def create_filename(self, name, item_type, **kwargs):
        pass

    def create_filenames(self, name, item_types, **kwargs):
        get_file = self.create_filename
        return {key: get_file(name, key, **kwargs) for key in item_types}


class TypeOnlyFormatter(FilenameFormatter):

    def create_filename(self, name, item_type, **kwargs):
        return f'{item_type}'


class NameWithTypeSuffixFormatter(FilenameFormatter):

    def create_filename(self, name, item_key, **kwargs):
        delim = kwargs.get('delimiter') or '_'
        return f'{name}{delim}{item_key}'
