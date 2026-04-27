r"""Logging Configuration Utilities.

This module provides custom formatting and initialization for Loguru-based 
logging. It includes level-aware message formatting (icons, colors, and 
metadata) and seamless integration with Python's standard `warnings` module.
"""

import sys
import warnings

from loguru import logger


def get_message_format(with_mp: bool = False):
    """Generates a dynamic, level-aware log message format.

    This function returns a callable that Loguru uses to format each record.
    The format adapts based on the severity level:
    - **DEBUG**: Shows function names in green.
    - **WARNING**: Highlights the message in yellow.
    - **INFO/SUCCESS**: Clean output with level icons.
    - **Exception**: Automatically appends stack traces if present.

    Args:
        with_mp: If True, includes the process name in the log prefix. 
            Useful for multiprocessing debugging.

    Returns:
        Callable: A function that takes a `record` and returns a format string.
    """

    def level_aware_format(record):
        # see: https://loguru.readthedocs.io/en/stable/api/logger.html#record
        level_icon = "<lvl>{level.icon} {level:<7}</lvl>"
        separator = " <c>|></c> "
        process_info = "<magenta>{process.name:<12}</magenta> " if with_mp else ""
        msg_format = level_icon + process_info + separator
        if record.get("level").name == "DEBUG":
            msg_format += "<g>({function})</g> {message}"
        elif record.get("level").name == "WARNING":
            level_icon = "<lvl>{level.icon}  {level:<7}</lvl>"
            msg_format = level_icon + process_info + separator + "<g>({function})</g> <y>{message}</y>"
        elif record.get("level").name == "SUCCESS":
            msg_format += "<g>{message}</g>"
        elif record.get("level").name == "INFO":
            level_icon = "<lvl>{level.icon}  {level:<7}</lvl>"
            msg_format = level_icon + process_info + separator + "{message}"
        else:
            msg_format += "{message}"
        return msg_format + "\n{exception}"
    return level_aware_format


def set_logger(enable: bool = True, capture_warnings: bool = True, **kwargs):
    """Configures and activates the global logger for the library.

    This is the main entry point to enable logging in `pysparta`. It removes 
    the default Loguru handler and sets up a customized one pointing to 
    `sys.stderr`.

    Args:
        enable: If False, disables all logs for the 'pysparta' namespace.
        capture_warnings: If True, redirects Python's `warnings.showwarning` 
            to the logger, ensuring consistency across all alerts.
        **kwargs: Additional arguments passed directly to `logger.add()`, 
            allowing overrides of `sink`, `level`, `format`, etc.

    Example:
        >>> from pysparta.logtools import set_logger
        >>> set_logger(level="DEBUG", with_mp=True)
    """

    if enable:
        default_kwargs = dict(
            sink=sys.stderr,
            level="INFO",
            format=get_message_format(),
            colorize=True,
        )
        logger.remove()
        logger.add(**(default_kwargs | (kwargs or {})))
        logger.enable("pysparta")
        if capture_warnings:
            def loguru_shows_warnings(message, category, filename, lineno, *args, **kwargs):
                logger.opt(depth=2).warning(f"{category.__name__}: {message}")
            warnings.showwarning = loguru_shows_warnings
    else:
        logger.disable("pysparta")


