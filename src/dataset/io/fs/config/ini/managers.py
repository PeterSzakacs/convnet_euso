import configparser
import os
import typing as t

from . import base
from . import versions
from .. import base as config_base


class IniConfigPersistenceManager(
    config_base.SingleFileConfigPersistenceManager
):

    def __init__(
            self,
            config_converter: base.AbstractIniConfigConverter = None
    ):
        """Base class for managing persistence of dataset properties/attributes
        using config in INI format.

        The actual logic for conversion of dataset properties/attributes to and
        from the filesystem stored config format is handled by separate classes
        implementing the AbstractIniConfigConverter interface to which this
        class delegates. This is to enable versioning and a gradual evolution
        of the config structure.

        :param config_converter: Converter instance to use by default for
                                 saving all managed configs during this manager
                                 instance lifecycle (defaults to the one for
                                 the highest current version of the INI config
                                 format)
        """
        # default/highest config version is 0
        self.config_converter = (config_converter
                                 or versions.get_ini_converter(0))

    # properties

    @property
    def config_converter(self):
        return self._converter

    @config_converter.setter
    def config_converter(self, value):
        self._converter = value

    # methods

    def get_config_version(
            self,
            file: str
    ) -> int:
        """Get dataset config version from the given config file.

        :param file: Dataset config file (in INI format)
        :return: Config version as an int
        """
        parser = self._read_config(file)
        try:
            # The 'general' section and its 'version' attribute shall be until
            # further notice mandatory for all config versions going forward.
            # Therefore any converter should be able
            return int(parser['general']['version'])
        except KeyError:
            return 0

    def load(
            self,
            file: str
    ) -> t.Mapping[str, t.Any]:
        """Load dataset attributes/properties from INI formatted config file.

        :param file: Dataset config filename/path
        :return: Representation of dataset attributes/properties as a dict or
                 mapping
        """
        parser = self._read_config(file)
        converter = self.config_converter
        version = self._get_version(parser)
        if version != converter.version:
            converter = versions.get_ini_converter(version)

        raw_config = {s: dict(parser.items(s)) for s in parser.sections()}
        return converter.parse_config(raw_config)

    def save(
            self,
            file: str,
            dataset_attrs: t.Mapping[str, t.Any]
    ) -> None:
        """Save dataset attributes/properties as a single INI-formatted config
        file with the given name.

        :param file: Filename for storing dataset config
        :param dataset_attrs: Dataset attributes/properties from which the
                              INI-formatted config representing them is created
        """
        raw_config = self.config_converter.create_config(dataset_attrs)
        parser = configparser.ConfigParser()
        parser.update(raw_config)
        with open(file, 'w', encoding='UTF-8') as configfile:
            parser.write(configfile)

    # helper methods

    @staticmethod
    def _read_config(
            file: str
    ):
        if not os.path.isfile(file):
            raise FileNotFoundError(
                'Config file {} does not exist'.format(file))
        parser = configparser.ConfigParser()
        parser.read(file, encoding='UTF-8')
        return parser

    @staticmethod
    def _get_version(
            parser: configparser.ConfigParser
    ):
        try:
            # The 'general' section and its 'version' attribute shall be until
            # further notice mandatory for all config versions going forward.
            # For legacy configs without such an attribute, it defaults to 0,
            # since the version 0 parser is specifically built to handle these.
            return int(parser['general']['version'])
        except KeyError:
            return 0
