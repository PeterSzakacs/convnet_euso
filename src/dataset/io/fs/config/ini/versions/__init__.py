import importlib
import typing as t

from .. import base


def get_ini_converter(
        version: t.Union[t.SupportsInt, str]
) -> base.AbstractIniConfigConverter:
    """
    Get INI config parser to handle dataset configs of a given version.

    :param version: INI config version
    :return: parser instance supporting given config version
    """
    module_name = f"{__package__}.version{int(version)}"
    module = importlib.import_module(module_name)
    cls = getattr(module, "ConfigConverter")
    converter = cls()
    return converter
