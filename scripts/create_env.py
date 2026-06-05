
import getpass
import os
import sys
from pathlib import Path

from loguru import logger
from pykeepass import PyKeePass

logger.remove()  # Remove the default handler.
logger.add(
    sink=sys.stderr,
    level="INFO",
    format="<lvl>{level}:</lvl> <lvl>{message}</lvl>",
    colorize=True)
logger = logger.opt(colors=True)


def get_envvar_path(env_var: str) -> str:
    """Get an environment variable, exiting with an error if not set.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to retrieve.
    Returns
    -------
    str
        The value of the environment variable.
    """
    if not (env_var_file := os.environ.get(env_var)):
        raise ValueError(f"'{env_var}' not set")

    if not (env_var_file := Path(env_var_file).expanduser()).is_file():
        raise ValueError(f"'{env_var_file}' does not exist")

    return env_var_file

try:
    KEEPASS_DB_FILE = get_envvar_path("KEEPASS_DB_FILE")
except ValueError as e:
    logger.error(e)
    exit(1)

logger.info(f"Using Keepass DB <blue>{KEEPASS_DB_FILE}</blue>")

try:
    KEEPASS_KEY_FILE = get_envvar_path("KEEPASS_KEY_FILE")
except ValueError as e:
    logger.error(e)
    exit(1)

logger.info(f"Using Keepass key file <blue>{KEEPASS_KEY_FILE}</blue>")

master_password = getpass.getpass("Introduce contraseña: ")

with PyKeePass(KEEPASS_DB_FILE, password=master_password, keyfile=KEEPASS_KEY_FILE) as kp:
    entries = kp.find_entries(title="pypi")
    if entries:
        entry = entries[0]
        print(f"Found entry: {entry.title}")
    else:
        print("Entry not found.")