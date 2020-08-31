import abc
import importlib


def get_ini_parser(version):
    """
    Get INI config parser to handle dataset configs of a given version.

    :param version: INI config version
    :type version: int or str
    :return: parser instance supporting given config version
    :type: AbstractConfigParser
    """
    module_name = f"dataset.io.fs.config.ini.versions.version{int(version)}"
    module = importlib.import_module(module_name)
    cls = getattr(module, "ConfigParser")
    parser = cls()
    return parser


class AbstractIniConfigParser(abc.ABC):

    @property
    @abc.abstractmethod
    def version(self):
        """
        Config version which this parser supports
        """
        pass

    @abc.abstractmethod
    def parse_config(self, raw_config):
        """
        Parse INI config dict as loaded by ConfigParser. The layout and
        attributes of the supported INI configs handled is version-dependent.
        """
        pass

    @abc.abstractmethod
    def create_config(self, dataset_attributes):
        """
        Create INI config dict for storage by ConfigParser. The layout and
        attributes of the produced INI config is version-dependent.
        """
        pass
