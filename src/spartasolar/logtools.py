
"""Logging format helpers used by spartasolar logging setup."""

import sys

from loguru import logger


def get_message_format(with_mp: bool = False):
    """Build a loguru format callable with level-aware styling.

    Parameters
    ----------
    with_mp : bool, default False
        If ``True``, include process name to help identify multi-process logs.

    Returns
    -------
    Callable[[dict], str]
        Formatter callable compatible with ``logger.add(format=...)``.
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

def filtrar_logs(filtros: list[str] | None = None):
    """Create a filter function to suppress noisy namespaces.

    Parameters
    ----------
    filtros : list[str] or None, default None
        Module name prefixes to filter. Matching records are only kept when
        severity is ERROR or higher.

    Returns
    -------
    Callable[[dict], bool]
        Predicate suitable for ``logger.add(filter=...)``.
    """
    def filtro(record):
        if any([record["name"].startswith(name) for name in filtros or []]):
            return record["level"].no >= logger.level("ERROR").no  # only pass if is an ERROR or more severe
        return True  # any other will pass through
    return filtro

def enable_logger(name: str | None = None, with_mp: bool = False, filtros: list[str] | None = None, **kwargs):
    """Configure and enable package logging.

    Parameters
    ----------
    name : str or None, default None
        Logger namespace to enable explicitly. If ``None``, enables ``__main__``.
    with_mp : bool, default False
        Enable multi-process-friendly logging options.
    filtros : list[str] or None, default None
        Prefixes to filter via :func:`filtrar_logs`.
    **kwargs : Any
        Extra keyword arguments forwarded to ``logger.add``.

    Notes
    -----
    Existing handlers are removed before installing the new one.

    Examples
    --------
    >>> from spartasolar.logtools import enable_logger
    >>> enable_logger("spartasolar", level="INFO")
    """
    global logger
    logger.remove()  # Remove the default handler.
    default_kwargs = dict(
        sink=sys.stderr,
        level="INFO",
        format=get_message_format(with_mp=with_mp),
        colorize=True,
        enqueue=with_mp,
        filter=filtrar_logs(filtros or []),
    )
    logger.add(**(default_kwargs | (kwargs or {})))
    logger = logger.opt(colors=True)
    logger.enable(name or "__main__")
    logger.enable("spartasolar")

def disable_logger():
    """Disable logging for the spartasolar namespace."""
    logger.disable("spartasolar")