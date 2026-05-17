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
    
    This validator checks if a string matches a specified regex pattern.
    Optionally, a parser function can transform the validated string before
    returning it.

    Args:
        pattern: The regex pattern to match against. Use raw strings (r"...") 
            for patterns containing backslashes.
        parser: An optional callable to transform the value after validation.
            The function receives the validated string and returns the transformed result.

    Examples:
        >>> # Validate email addresses
        >>> email_validator = ValidaRegex(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        >>> email_validator.validate("user@example.com")
        'user@example.com'
        
        >>> # Validate and uppercase
        >>> code_validator = ValidaRegex(pattern=r"^[A-Z]{3}$", parser=str.upper)
        >>> code_validator.validate("abc")  # Will be converted to uppercase
        'ABC'
        
        >>> # Invalid input
        >>> email_validator.validate("not-an-email")  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: not-an-email must match the regex pattern: ...
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

    This validator performs case-insensitive matching against a list of allowed
    values. If an exact match (ignoring case) is not found, it attempts fuzzy 
    matching to correct potential typos. A warning is issued when fuzzy matching
    is used.

    Args:
        choices: List of allowed string values (canonical forms).
        parser: An optional callable to transform the value after validation.

    Notes:
        - Matching is case-insensitive: "sparta", "SPARTA", and "Sparta" all match.
        - Fuzzy matching uses a similarity threshold of 0.4 (40% similarity).
        - The returned value is always in the canonical form from the choices list.

    Examples:
        >>> # Validate model names
        >>> model_validator = ValidaChoices(choices=["SPARTA", "BIRD"])
        >>> model_validator.validate("sparta")  # Case-insensitive
        'SPARTA'
        
        >>> model_validator.validate("BIRD")
        'BIRD'
        
        >>> # Fuzzy matching with warning
        >>> model_validator.validate("sprata")  # doctest: +SKIP
        UserWarning: sprata does not match the allowed choices. I took the closest one: SPARTA
        'SPARTA'
        
        >>> # No match found
        >>> model_validator.validate("INVALID")  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: INVALID is not among the allowed choices: ['SPARTA', 'BIRD']
        
        >>> # With parser
        >>> uppercase_validator = ValidaChoices(choices=["one", "two"], parser=str.upper)
        >>> uppercase_validator.validate("one")
        'ONE'
    """
    choices: list[str]
    parser: Callable[[str], str] | None = None

    def validate(self, value: str) -> str:
        """Validates a string against the allowed choices.

        Args:
            value: The string to validate.

        Returns:
            The matched choice in canonical form (corrected via fuzzy matching if necessary).

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

    Validates that numeric values fall within specified boundaries using
    inclusive (ge/le) or exclusive (gt/lt) comparisons. Multiple constraints
    can be combined to define precise ranges.

    Args:
        le: Maximum value (inclusive). Value must be ≤ this limit.
        lt: Maximum value (exclusive). Value must be < this limit.
        ge: Minimum value (inclusive). Value must be ≥ this limit.
        gt: Minimum value (exclusive). Value must be > this limit.
        parser: An optional callable to transform the value after validation.

    Notes:
        - String inputs are automatically converted to float.
        - Only one upper bound (le or lt) should be specified.
        - Only one lower bound (ge or gt) should be specified.
        - Use ge/le for closed intervals: [min, max]
        - Use gt/lt for open intervals: (min, max)

    Examples:
        >>> # Validate percentage (0 to 100, inclusive)
        >>> percentage = ValidaRange(ge=0, le=100)
        >>> percentage.validate(50)
        50.0
        >>> percentage.validate(100)
        100.0
        
        >>> # Validate latitude (-90 < lat < 90, exclusive)
        >>> latitude = ValidaRange(gt=-90, lt=90)
        >>> latitude.validate(45.5)
        45.5
        
        >>> # String conversion
        >>> percentage.validate("75.5")
        75.5
        
        >>> # Out of range
        >>> percentage.validate(101)  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: 101 must be less or equal than 100
        
        >>> # With parser (double the value)
        >>> doubler = ValidaRange(ge=0, le=50, parser=lambda x: x * 2)
        >>> doubler.validate(25)
        50.0
    """
    le: float | None = None  # less or equal than this
    lt: float | None = None  # less than this
    ge: float | None = None  # greater or equal than this
    gt: float | None = None  # greater than this
    parser: Callable[[float], float] | None = None

    def validate(self, value: float | int | str ) -> float:
        """Validates that a number falls within the specified range.

        Args:
            value: The numerical value to validate. Strings are converted to float.

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

    This function is the main entry point for type validation. It extracts the
    validator from an `Annotated` type alias and executes its `validate` method.
    
    This enables declarative type validation using Python's type hints system.

    Args:
        value: The value to be validated. Can be of any type.
        annotated_type: A type alias defined with `Annotated[base_type, Validator(...)]`.
            Must be created using the `type` statement with an `Annotated` type.

    Returns:
        The validated (and possibly transformed) value. Returns None if the input 
        value is None (allows for optional parameters).

    Raises:
        TypeError: If the provided `annotated_type` is not a valid `Annotated` type.
        ValueError: If validation fails (specific message depends on the validator).

    Examples:
        >>> # Using predefined type aliases
        >>> validate_type(40.4, Latitude)
        40.4
        
        >>> validate_type(-3.7, Longitude)
        -3.7
        
        >>> # Model name validation (case-insensitive)
        >>> validate_type("sparta", Model)
        'SPARTA'
        
        >>> # None values pass through
        >>> validate_type(None, Latitude) is None
        True
        
        >>> # Invalid latitude
        >>> validate_type(95, Latitude)  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: 95 must be less than 90
        
        >>> # Custom type alias
        >>> type Percentage = Annotated[float, ValidaRange(ge=0, le=100)]
        >>> validate_type(75.5, Percentage)
        75.5

    See Also:
        - Latitude: Validates latitude coordinates (-90° < lat < 90°)
        - Longitude: Validates longitude coordinates (-180° ≤ lon < 180°)
        - Elevation: Validates elevation/altitude (-450m < elev < 8900m)
        - Model: Validates model names (SPARTA, BIRD)
    """
    if value is not None:
        anntype_value = annotated_type.__value__
        if not hasattr(anntype_value, "__origin__") or get_origin(anntype_value) is not Annotated:
            raise TypeError(f"{annotated_type} is not an Annotated type")
        _, validator = get_args(anntype_value)
        return validator.validate(value)
    return None


type Latitude = Annotated[float, ValidaRange(gt=-90, lt=90)]
"""Geographic latitude coordinate validator.

Validates latitude values in decimal degrees. Range: -90° < lat < 90° (exclusive).

Examples:
    >>> from spartasolar.validation import validate_type, Latitude
    >>> validate_type(40.4168, Latitude)  # Madrid
    40.4168
    >>> validate_type(-33.8688, Latitude)  # Sydney
    -33.8688
"""

type Longitude = Annotated[float, ValidaRange(ge=-180, lt=180)]
"""Geographic longitude coordinate validator.

Validates longitude values in decimal degrees. Range: -180° ≤ lon < 180°.

Examples:
    >>> from spartasolar.validation import validate_type, Longitude
    >>> validate_type(-3.7038, Longitude)  # Madrid
    -3.7038
    >>> validate_type(151.2093, Longitude)  # Sydney
    151.2093
"""

type Elevation = Annotated[float, ValidaRange(gt=-450, lt=8900)]
"""Surface elevation/altitude validator.

Validates elevation in meters above sea level. Range: -450m < elev < 8900m.
Covers from Dead Sea (-430m) to Mt. Everest (8849m).

Examples:
    >>> from spartasolar.validation import validate_type, Elevation
    >>> validate_type(667, Elevation)  # Madrid
    667.0
    >>> validate_type(0, Elevation)  # Sea level
    0.0
"""

type SodaTimeStep = Annotated[str, ValidaChoices(["PT01M", "PT15M", "PT01H", "PT01D", "P01M"])]
"""Temporal resolution for SoDA API requests.

Allowed values (ISO 8601 duration format):
    - `PT01M`: 1 minute
    - `PT15M`: 15 minutes  
    - `PT01H`: 1 hour (hourly)
    - `PT01D`: 1 day (daily)
    - `P01M`: 1 month (monthly)

Examples:
    >>> from spartasolar.validation import validate_type, SodaTimeStep
    >>> validate_type("PT01H", SodaTimeStep)
    'PT01H'
"""

type SodaStream = Annotated[str, ValidaChoices(["mcclear", "cams_radiation"])]
"""Available data streams from the SoDA service.

Allowed values:
    - `mcclear`: McClear clear-sky irradiation model
    - `cams_radiation`: CAMS all-sky radiation service

Examples:
    >>> from spartasolar.validation import validate_type, SodaStream
    >>> validate_type("mcclear", SodaStream)
    'mcclear'
"""

type Model = Annotated[str, ValidaChoices(["SPARTA", "BIRD"])]
"""Solar radiation model name validator.

Allowed values:
    - `SPARTA`: Solar PArameterization for the Radiative Transfer of the Atmosphere
    - `BIRD`: Bird clear-sky model

Note:
    Validation is case-insensitive. Input will be converted to uppercase.

Examples:
    >>> from spartasolar.validation import validate_type, Model
    >>> validate_type("sparta", Model)
    'SPARTA'
    >>> validate_type("BIRD", Model)
    'BIRD'
"""

type Atmosphere = Annotated[str, ValidaChoices(list(atmoslib.atmos_dict))]
