from .base import SingleFileConfigPersistenceManager
from .ini.managers import IniConfigPersistenceManager


def get_config_manager(
        config_format: str = 'ini'
) -> SingleFileConfigPersistenceManager:
    """Retrieve manager for file-backed dataset configs based on given criteria
    (currently only by config format, e.g. ini, xml, json, etc.).

    Currently supported values for config format:
    - ini - managed using python's built-in configparser library

    :param config_format: (case-insensitive) format of the managed dataset
                          config(s).
    :return: config manager for the given criteria
    :raise ValueError: if a manager implementation for the given criteria is
                       not found
    """
    _format = config_format.lower()
    if _format == 'ini':
        return IniConfigPersistenceManager()
    else:
        raise ValueError(f'Unsupported config format: {config_format}')
