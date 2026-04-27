r"""Configuration Management for Pysparta.

This module handles the persistent storage and retrieval of user settings using 
a TOML configuration file located in the standard user configuration directory.

It manages global library options such as API credentials (e.g., SODA's user email) 
and local storage paths (e.g., MERRA-2 daily data`data_dir`).
"""

# HOW DOES THIS CONFIG WORK? WHEN THE FLOW PASS THROUGH, IT CHECKS IF THE CONFIG
# FILE EXISTS PHYSICALLY IN LOCAL. IF NOT, IT CREATES IT WITH THE DEFAULT SET-UP.
# THEN, THE FILE IS READ AND STORED IN _GLOBAL_CONFIG.
# THE CONFIGURATION CAN BE CONSULTED WITH show_options AND get_option, AND EVEN
# MODIFIED WITH set_option. HOWEVER, CHANGES ONLY AFFECT THE CURRENT SESSION.
# TO PERSIST THE CHANGES THE USER MUST DO IT MANUALLY BY EDITING THE CONFIG FILE.
# THE PATH TO THE LOCAL FILE IS AVAILABLE IN get_config_path

from pathlib import Path
from typing import Any

import platformdirs
import tomlkit
from loguru import logger

logger.disable(__name__)
logger = logger.opt(colors=True)

_DEFAULT_CONFIG_TOML_="""
[crs_soda]
# # Table for the soda-api's CAMS Radiation Service (CRS).
# # Only for single locations.
# # The data are fetched for lat and lon truncated with 4 decimals
# # and archived per site in yearly chunks in tabular parquet files.
# user_email = "your@email"  # soda-api user's email that grants you access the data
# data_dir = "<your-optional-choice>"  # optional local path to cache the data

[merra2_gee]
# # Table for the Google Earth Engine's MERRA-2 hourly data.
# # Only for single locations.
# # The data are fetched for lat and lon truncated with 4 decimals
# # and archived per site in yearly chunks in tabular parquet files.
# project = "<your-google-cloud-project>"  # your Google cloud's project to retrieve data
# data_dir = "<your-optional-choice>"  # optional local path to cache the data

[merra2_daily]
# Table for the MERRA-2 daily data shared in Hugging Face Hub.
# data_dir = "<your-optional-choice>"  # optional local path to cache the data

[sunwhere]  # table to set the sunwhere's options
algorithm = "psa"  # solar position algorithm
refraction = true
engine = "numexpr"
"""

def get_config_path() -> Path:
    """Get the path to the user's configuration file.

    Returns:
        Path: The absolute path to `config.toml` within the standard 
            system-specific user configuration directory.
    """
    path = platformdirs.user_config_path(appname="pysparta", ensure_exists=True)
    return path / "config.toml"

def _init_config_file():
    with get_config_path().open(mode="w") as f:
        f.write(_DEFAULT_CONFIG_TOML_)
    logger.success(f"user's config file initialized at <blue>{get_config_path()}</blue>")

def _read_config_options() -> dict[str, Any]:
    """Read all options from the configuration file.

    If the configuration file does not exist, it initializes it with 
    default placeholder values.

    Returns:
        dict[str, Any]: A dictionary containing the configuration keys 
            and their values parsed from the TOML file.
    """

    if not (config_path := get_config_path()).exists():
        _init_config_file()

    with config_path.open(mode="rb") as f:
        return tomlkit.load(f)

def _reset_config_file():
    global _GLOBAL_CONFIG
    if get_config_path().exists():
        get_config_path().unlink()
        logger.success(f"config file {get_config_path()} deleted")
    _GLOBAL_CONFIG = _read_config_options()

_GLOBAL_CONFIG = _read_config_options()

def show_config() -> None:
    """Print all current global options to the console.

    Note:
        This function uses `pprint` for a formatted output of the 
        global configuration state.
    """
    from pprint import pprint
    return pprint(_GLOBAL_CONFIG, indent=2, width=20)

def get_option(name: str, default: Any = None) -> Any:
    """Retrieve the value of a specific configuration option.

    Args:
        name: The name of the option to retrieve using the format
            <table-name>.<option-name> (e.g., 'crs_soda.user_email').

    Returns:
        Any: The value of the option. Returns `None` if the option 
            is missing or set to the placeholder "__NOTSET__". 
            Special case: 'data_dir' is always returned as a `Path` object.
    """
    table_name, option_name = name.split(".")
    if (table := _GLOBAL_CONFIG.get(table_name, None)) is None:
        logger.warning(f"missing table `{table_name}`")
        return None
    if (value := table.get(option_name, None)) is None:
        return default
    if option_name == "data_dir":
        return Path(value)
    return value

def set_option(name: str, value: Any) -> None:
    """Temporarily update a global option for the current session.

    Args:
        name: The name of the option to update.
        value: The new value to assign.

    Note:
        This update only affects the current runtime session. It does not 
        persist the value to the `config.toml` file.
    """

    table_name, option_name = name.split(".")
    if _GLOBAL_CONFIG.get(table_name, None) is None:
        logger.warning(f"missing table `{table_name}`")
        return None
    if option_name == "data_dir" and isinstance(value, Path):
        value = value.as_posix()
    _GLOBAL_CONFIG[table_name][option_name] = value
