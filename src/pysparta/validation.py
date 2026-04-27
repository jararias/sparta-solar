r"""Type Validation and Custom Annotated Types.

This module provides a framework for robust type validation using Python's 
`Annotated` types and custom validator classes. It includes validators for 
regular expressions, fixed choices (with fuzzy matching), and numerical ranges.

The module also defines several domain-specific types for solar geometry 
and atmospheric data (Latitude, Longitude, SodaStream, etc.).
"""

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Annotated, Any, get_args, get_origin

from . import atmoslib

# from loguru import logger


@dataclass
class ValidaRegex:
    """Validator for string patterns using regular expressions.

    Args:
        pattern: The regex pattern to match against.
        parser: An optional callable to transform the value after validation.
    """
    pattern: str
    parser: Callable[[str], str] | None = None

    def validate(self, value: str) -> str:
        """Validates a string against the regex pattern.

        Args:
            value: The string to validate.

        Returns:
            The (potentially parsed) string if validation succeeds.

        Raises:
            TypeError: If the value is not a string.
            ValueError: If the value does not match the pattern.
        """
        if not isinstance(value, str):
            raise TypeError(f"{value} must be a string")
        if not re.match(self.pattern, value):
            raise ValueError(f"{value} must match the regex pattern: {self.pattern}")
        if self.parser is not None:
            return self.parser(value)
        return value

@dataclass
class ValidaChoices:
    """Validator for a fixed set of string choices with fuzzy matching.

    If a value does not match exactly but is close to an allowed choice, 
    a warning is issued and the closest match is used.

    Args:
        choices: List of allowed string values.
        parser: An optional callable to transform the value after validation.
    """
    choices: list[str]
    parser: Callable[[str], str] | None = None

    def validate(self, value: str) -> str:
        """Validates a string against the allowed choices.

        Args:
            value: The string to validate.

        Returns:
            The matched choice (corrected via fuzzy matching if necessary).

        Raises:
            TypeError: If the value is not a string.
            ValueError: If no close match is found among the choices.
        """
        if not isinstance(value, str):
            raise TypeError(f"{value} must be a string")
        case_safe_map = {choice.casefold(): choice for choice in self.choices}
        if value.casefold() not in case_safe_map:
            if not (matches := get_close_matches(value.casefold(), case_safe_map, n=1, cutoff=0.4)):
                raise ValueError(f"{value} is not among the allowed choices: {self.choices}")
            best_choice = case_safe_map[matches[0]]
            warnings.warn(f"{value} does not match the allowed choices. I took the closest one: {best_choice}")
            value = best_choice
        else:
            # Return the canonical value from choices, not the user input
            value = case_safe_map[value.casefold()]
        if self.parser is not None:
            return self.parser(value)
        return value

@dataclass
class ValidaRange:
    """Validator for numerical ranges.

    Supports inclusive and exclusive boundaries for float or integer values.

    Args:
        le: Less than or equal to.
        lt: Less than.
        ge: Greater than or equal to.
        gt: Greater than.
        parser: An optional callable to transform the value after validation.
    """
    le: float | None = None  # less or equal than this
    lt: float | None = None  # less than this
    ge: float | None = None  # greater or equal than this
    gt: float | None = None  # greater than this
    parser: Callable[[float], float] | None = None

    def validate(self, value: float | int | str ) -> float:
        """Validates that a number falls within the specified range.

        Args:
            value: The numerical value to validate.

        Returns:
            The (potentially parsed) float value.

        Raises:
            TypeError: If the value cannot be converted to a float.
            ValueError: If the value violates any of the range constraints.
        """
        try:
            value = float(value)
        except Exception:
            raise TypeError(f"{value} must be a number")
        if (self.le is not None) and (value > self.le):
            raise ValueError(f"{value} must be less or equal than {self.le}")
        if (self.lt is not None) and (value >= self.lt):
            raise ValueError(f"{value} must be less than {self.lt}")
        if (self.ge is not None) and (value < self.ge):
            raise ValueError(f"{value} must be greater or equal than {self.ge}")
        if (self.gt is not None) and (value <= self.gt):
            raise ValueError(f"{value} must be greater than {self.gt}")
        if self.parser is not None:
            return self.parser(value)
        return value

def validate_type(value: Any, annotated_type: Any) -> Any:
    """Validates a value against an Annotated type definition.

    This function extracts the validator from an `Annotated` type and 
    executes its `validate` method.

    Args:
        value: The value to be checked.
        annotated_type: The type alias defined with `Annotated`.

    Returns:
        The validated (and possibly transformed) value. 
        Returns None if the input value is None.

    Raises:
        TypeError: If the provided `annotated_type` is not a valid 
            `Annotated` object.
    """
    if value is not None:
        anntype_value = annotated_type.__value__
        if not hasattr(anntype_value, "__origin__") or get_origin(anntype_value) is not Annotated:
            raise TypeError(f"{annotated_type} is not an Annotated type")
        _, validator = get_args(anntype_value)
        return validator.validate(value)
    return None


type Latitude = Annotated[float, ValidaRange(gt=-90, lt=90)]

type Longitude = Annotated[float, ValidaRange(ge=-180, lt=180)]

type Elevation = Annotated[float, ValidaRange(gt=-450, lt=8900)]

type SodaTimeStep = Annotated[str, ValidaChoices(["PT01M", "PT15M", "PT01H", "PT01D", "P01M"])]
"""Temporal resolution for SoDA API requests.

Allowed values: `PT01M` (1 minute), `PT15M` (15 minutes), `PT01H` (hourly), `PT01D` (daily) and `P01M` (monthly).
"""

type SodaStream = Annotated[str, ValidaChoices(["mcclear", "cams_radiation"])]
"""Available data streams from the SoDA service.

Allowed values: `mcclear` for clear-sky irradiation data and `cams_radiation` for all-sky conditions.
"""

type Model = Annotated[str, ValidaChoices(["SPARTA", "BIRD"])]

type Atmosphere = Annotated[str, ValidaChoices(list(atmoslib.atmos_dict))]
