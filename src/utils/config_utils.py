import configparser
import os
import pathlib


def get_config_for_module(module_name):
    filename = "{}.config.ini".format(module_name)
    config_pathname = os.path.join(str(pathlib.Path.home()), filename)
    cparser = configparser.ConfigParser(os.environ)
    cparser.read(config_pathname)
    return cparser
