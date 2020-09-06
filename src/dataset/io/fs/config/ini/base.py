import abc
import importlib
import typing as t


def get_ini_converter(version):
    """
    Get INI config parser to handle dataset configs of a given version.

    :param version: INI config version
    :type version: int or str
    :return: parser instance supporting given config version
    :type: AbstractConfigParser
    """
    module_name = f"dataset.io.fs.config.ini.versions.version{int(version)}"
    module = importlib.import_module(module_name)
    cls = getattr(module, "ConfigConverter")
    parser = cls()
    return parser


class AbstractIniConfigConverter(abc.ABC):

    @property
    @abc.abstractmethod
    def version(self) -> int:
        """
        Config version which this converter supports.
        """
        pass

    @abc.abstractmethod
    def parse_config(
            self,
            raw_config: t.Mapping[str, t.Any]
    ) -> t.Mapping[str, t.Any]:
        """
        Parse INI config dict as loaded by configparser.ConfigParser. Layout
        and attributes of the INI configs handled by a given parser class is
        version-dependent.

        :param raw_config: INI-formatted dataset config as dict/Mapping, with
                           keys representing section and attribute names and
                           values their respective values (for sections, the
                           value is always a nested dict).
        :returns: format-independent dict representation of dataset attributes
                  or properties
        """
        pass

    @abc.abstractmethod
    def create_config(
            self,
            dataset_attributes: t.Mapping[str, t.Any]
    ) -> t.Mapping[str, t.Any]:
        """
        Create INI config dict for storage by ConfigParser. The layout and
        attributes of the produced INI config is version-dependent.

        :param dataset_attributes: format-independent dict representation of
                                   dataset attributes/properties
        :returns: INI-formatted dataset config as dict/Mapping, with keys
                  representing section and attribute names and values their
                  respective values (for sections, the value is always a nested
                  dict).
        """
        pass
