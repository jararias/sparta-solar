r"""Configuration Management for SPARTA-Solar.

This module handles the persistent storage and retrieval of user settings using 
a TOML configuration file located in the standard user configuration directory.

The configuration system manages:
    - API credentials (e.g., SODA's user email)
    - Local storage paths for cached data
    - Algorithm preferences (e.g., solar position algorithm)
    - Service-specific settings

Configuration Flow:
    1. On first import, checks if config file exists
    2. If not found, creates it with default template
    3. Loads configuration into memory (_GLOBAL_CONFIG)
    4. Changes via set_option() are session-only
    5. Persistent changes require manual editing of config.toml

Configuration File Location:
    - Linux: ~/.config/spartasolar/config.toml
    - macOS: ~/Library/Application Support/spartasolar/config.toml
    - Windows: C:\\Users\\<user>\\AppData\\Local\\spartasolar\\config.toml

Examples:
    >>> from spartasolar.config import get_config_path, get_option, set_option
    
    >>> # Get configuration file path
    >>> config_path = get_config_path()
    >>> print(config_path)
    PosixPath('/home/user/.config/spartasolar/config.toml')
    
    >>> # Retrieve an option
    >>> email = get_option('crs_soda.user_email')
    
    >>> # Set option for current session only
    >>> set_option('sunwhere.algorithm', 'spa')
    
    >>> # Get data directory (returns Path object)
    >>> data_dir = get_option('merra2_daily.data_dir')

See Also:
    - get_config_path(): Get path to configuration file
    - get_option(): Retrieve configuration value
    - set_option(): Modify configuration (session only)
    - show_config(): Display all current settings
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
    path = platformdirs.user_config_path(appname="sparta-solar", ensure_exists=True)
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
    
    Options are organized in tables (sections) within the TOML file.
    This function uses dot notation to access nested values.

    Args:
        name: The name of the option to retrieve using the format
            `<table-name>.<option-name>` (e.g., 'crs_soda.user_email').
        default: Value to return if the option is not found. Defaults to None.

    Returns:
        Any: The value of the option. Returns `default` if the option 
            is missing. Special case: options named 'data_dir' are 
            automatically converted to `Path` objects.
            
    Examples:
        >>> from spartasolar.config import get_option
        
        >>> # Get solar position algorithm
        >>> algorithm = get_option('sunwhere.algorithm')
        >>> print(algorithm)
        'psa'
        
        >>> # Get with default value
        >>> email = get_option('crs_soda.user_email', default='user@example.com')
        
        >>> # Data directories return Path objects
        >>> from pathlib import Path
        >>> data_dir = get_option('merra2_daily.data_dir')
        >>> isinstance(data_dir, (Path, type(None)))
        True
    """
    table_name, option_name = name.split(".")
    if (table := _GLOBAL_CONFIG.get(table_name, None)) is None:
        logger.warning(f"missing table `{table_name}`")
        return default
    if (value := table.get(option_name, None)) is None:
        return default
    if option_name == "data_dir":
        return Path(value)
    return value

def set_option(name: str, value: Any) -> None:
    """Temporarily update a global option for the current session.
    
    Modifies configuration values in memory only. Changes are lost when
    the Python session ends. To make persistent changes, edit the
    config.toml file directly.

    Args:
        name: The name of the option to update in format `<table>.<option>`.
        value: The new value to assign. Path objects for 'data_dir' options
            are automatically converted to strings.

    Returns:
        None

    Warning:
        Session-only changes are NOT saved to the config file. Restart
        the Python session to revert to file values.
        
    Examples:
        >>> from spartasolar.config import set_option, get_option
        
        >>> # Change solar position algorithm
        >>> set_option('sunwhere.algorithm', 'spa')
        >>> get_option('sunwhere.algorithm')
        'spa'
        
        >>> # Set data directory with Path object
        >>> from pathlib import Path
        >>> set_option('merra2_daily.data_dir', Path('/custom/path'))
        
    Note:
        To persist changes, manually edit the configuration file at the
        path returned by get_config_path().
    """

    table_name, option_name = name.split(".")
    if _GLOBAL_CONFIG.get(table_name, None) is None:
        logger.warning(f"missing table `{table_name}`")
        return None
    if option_name == "data_dir" and isinstance(value, Path):
        value = value.as_posix()
    _GLOBAL_CONFIG[table_name][option_name] = value
