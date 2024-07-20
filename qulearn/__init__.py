import os

import toml


def get_version():
    here = os.path.dirname(os.path.realpath(__file__))
    version_info = toml.load(os.path.join(here, "../pyproject.toml"))
    return version_info["tool"]["poetry"]["version"]


__version__ = get_version()
