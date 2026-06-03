
# ruff: noqa: F401 

import importlib.metadata

from . import config
from .logtools import enable_logger, disable_logger

try:
    __version__ = importlib.metadata.version("sparta-solar")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


enable_logger("spartasolar", level="INFO")
