import configparser
import os
import pathlib


def get_config_for_module(module_name):
    parser = configparser.ConfigParser(os.environ)
    filename = "{}.config.ini".format(module_name)
    # system config
    config_path = os.path.join("/etc/convnet_euso/", filename)
    parser.read(config_path)
    # user config
    config_path = os.path.join(str(pathlib.Path.home()), filename)
    parser.read(config_path)
    return parser
