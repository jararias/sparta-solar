
# ruff: noqa: F401 

import importlib.metadata

from loguru import logger

from . import config
from .logtools import set_logger

try:
    __version__ = importlib.metadata.version("sparta-solar")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# De manera estándar, los logs de loguru se deberían dejar bloqueados
# usando logger.disable("spartasolar") y que sea el usuario final el que
# decida si quiere mostrarlos o no (habilitarlos por defecto no es lo
# recomendado).

# La forma general que tendría el usuario de activarlos sería:

# import spartasolar
# from loguru import logger
# logger.enable("spartasolar")

# No obstante, el usuario puede elegir aplicar la configuración que yo
# aporto en la aplicación:

# import spartasolar
# spartasolar.set_logger(level="INFO")

logger.disable("spartasolar")
