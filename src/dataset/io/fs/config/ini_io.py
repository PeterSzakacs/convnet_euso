import configparser
import os

import dataset.io.fs.base as fs_io_base
import dataset.io.fs.config.ini.base as ini_base


class IniConfigPersistencyHandler(fs_io_base.FsPersistencyHandler):

    # static attributes

    DEFAULT_CONFIG_FILE_SUFFIX = '_config.ini'

    def __init__(self, config_parser=None, file_suffix=None,
                 load_dir=None, save_dir=None):
        super(self.__class__, self).__init__(load_dir, save_dir)
        # default parser is at version 0
        self.config_parser = config_parser or ini_base.get_ini_parser(0)
        self.file_suffix = file_suffix or self.DEFAULT_CONFIG_FILE_SUFFIX

    # properties

    @property
    def file_suffix(self):
        return self._suffix

    @file_suffix.setter
    def file_suffix(self, value):
        self._suffix = value

    @property
    def config_parser(self):
        return self._parser

    @config_parser.setter
    def config_parser(self, value):
        self._parser = value

    # methods

    def get_config_version(self, name):
        self._check_before_read()
        parser = self._read_config(name)
        try:
            return int(parser['general']['version'])
        except KeyError:
            return 0

    def load_config(self, name):
        self._check_before_read()
        parser = self._read_config(name)
        raw_config = {s: dict(parser.items(s)) for s in parser.sections()}
        return self.config_parser.parse_config(raw_config)

    def save_config(self, name, dataset_dict):
        self._check_before_write()
        raw_config = self.config_parser.create_config(dataset_dict)
        parser = configparser.ConfigParser()
        parser.update(raw_config)
        filename = '{}{}'.format(name, self._suffix)
        filepath = os.path.join(self.savedir, filename)
        with open(filepath, 'w', encoding='UTF-8') as configfile:
            parser.write(configfile)

    # helper methods
    
    def _read_config(self, name):
        filename = '{}{}'.format(name, self._suffix)
        filepath = os.path.join(self.loaddir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                'Config file {} does not exist'.format(filepath))
        parser = configparser.ConfigParser()
        parser.read(filepath, encoding='UTF-8')
        return parser
