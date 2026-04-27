"""Unit tests for pysparta.validation module.

Tests cover:
- ValidaRegex: Pattern validation and parsing
- ValidaChoices: Choice validation with fuzzy matching
- ValidaRange: Numerical range validation
- validate_type: Main validation function
- Type aliases: Latitude, Longitude, etc.
"""

import pytest
from typing import Annotated

from pysparta.validation import (
    ValidaRegex,
    ValidaChoices,
    ValidaRange,
    validate_type,
    Latitude,
    Longitude,
    Elevation,
    SodaTimeStep,
    SodaStream,
    Model,
)


class TestValidaRegex:
    """Test suite for ValidaRegex validator."""

    def test_valid_pattern(self):
        """Test that valid patterns pass validation."""
        validator = ValidaRegex(pattern=r"^\d{3}-\d{4}$")
        assert validator.validate("123-4567") == "123-4567"

    def test_invalid_pattern(self):
        """Test that invalid patterns raise ValueError."""
        validator = ValidaRegex(pattern=r"^\d{3}-\d{4}$")
        with pytest.raises(ValueError, match="must match the regex pattern"):
            validator.validate("abc-defg")

    def test_non_string_input(self):
        """Test that non-string inputs raise TypeError."""
        validator = ValidaRegex(pattern=r"^\d+$")
        with pytest.raises(TypeError, match="must be a string"):
            validator.validate(12345)

    def test_with_parser(self):
        """Test that parser is applied after validation."""
        validator = ValidaRegex(pattern=r"^\w+$", parser=str.upper)
        assert validator.validate("hello") == "HELLO"


class TestValidaChoices:
    """Test suite for ValidaChoices validator."""

    def test_exact_match(self):
        """Test exact matching of choices."""
        validator = ValidaChoices(choices=["apple", "banana", "cherry"])
        assert validator.validate("apple") == "apple"

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        validator = ValidaChoices(choices=["Apple", "Banana", "Cherry"])
        assert validator.validate("apple") == "Apple"

    def test_fuzzy_match(self):
        """Test fuzzy matching with warning."""
        validator = ValidaChoices(choices=["sparta", "bird"])
        with pytest.warns(UserWarning, match="does not match the allowed choices"):
            result = validator.validate("sprata")
        assert result == "sparta"

    def test_no_close_match(self):
        """Test that completely wrong values raise ValueError."""
        validator = ValidaChoices(choices=["apple", "banana", "cherry"])
        with pytest.raises(ValueError, match="is not among the allowed choices"):
            validator.validate("xyz123")

    def test_non_string_input(self):
        """Test that non-string inputs raise TypeError."""
        validator = ValidaChoices(choices=["one", "two", "three"])
        with pytest.raises(TypeError, match="must be a string"):
            validator.validate(123)

    def test_with_parser(self):
        """Test that parser is applied after validation."""
        validator = ValidaChoices(choices=["one", "two"], parser=str.upper)
        assert validator.validate("one") == "ONE"


class TestValidaRange:
    """Test suite for ValidaRange validator."""

    def test_within_range_le(self):
        """Test less than or equal constraint."""
        validator = ValidaRange(le=100)
        assert validator.validate(50) == 50.0
        assert validator.validate(100) == 100.0

    def test_outside_range_le(self):
        """Test violation of less than or equal constraint."""
        validator = ValidaRange(le=100)
        with pytest.raises(ValueError, match="must be less or equal than"):
            validator.validate(101)

    def test_within_range_lt(self):
        """Test less than constraint."""
        validator = ValidaRange(lt=100)
        assert validator.validate(99) == 99.0

    def test_outside_range_lt(self):
        """Test violation of less than constraint."""
        validator = ValidaRange(lt=100)
        with pytest.raises(ValueError, match="must be less than"):
            validator.validate(100)

    def test_within_range_ge(self):
        """Test greater than or equal constraint."""
        validator = ValidaRange(ge=0)
        assert validator.validate(0) == 0.0
        assert validator.validate(50) == 50.0

    def test_outside_range_ge(self):
        """Test violation of greater than or equal constraint."""
        validator = ValidaRange(ge=0)
        with pytest.raises(ValueError, match="must be greater or equal than"):
            validator.validate(-1)

    def test_within_range_gt(self):
        """Test greater than constraint."""
        validator = ValidaRange(gt=0)
        assert validator.validate(1) == 1.0

    def test_outside_range_gt(self):
        """Test violation of greater than constraint."""
        validator = ValidaRange(gt=0)
        with pytest.raises(ValueError, match="must be greater than"):
            validator.validate(0)

    def test_combined_constraints(self):
        """Test multiple constraints combined."""
        validator = ValidaRange(ge=0, le=100)
        assert validator.validate(50) == 50.0
        with pytest.raises(ValueError):
            validator.validate(-1)
        with pytest.raises(ValueError):
            validator.validate(101)

    def test_string_to_float_conversion(self):
        """Test automatic conversion from string to float."""
        validator = ValidaRange(ge=0, le=100)
        assert validator.validate("50.5") == 50.5

    def test_non_numeric_input(self):
        """Test that non-numeric inputs raise TypeError."""
        validator = ValidaRange(ge=0)
        with pytest.raises(TypeError, match="must be a number"):
            validator.validate("abc")

    def test_with_parser(self):
        """Test that parser is applied after validation."""
        validator = ValidaRange(ge=0, le=100, parser=lambda x: x * 2)
        assert validator.validate(25) == 50.0


class TestValidateType:
    """Test suite for validate_type function."""

    def test_validate_latitude(self):
        """Test latitude validation."""
        assert validate_type(45.5, Latitude) == 45.5
        assert validate_type(-45.5, Latitude) == -45.5

    def test_validate_latitude_out_of_range(self):
        """Test latitude validation with invalid values."""
        with pytest.raises(ValueError):
            validate_type(91, Latitude)
        with pytest.raises(ValueError):
            validate_type(-91, Latitude)

    def test_validate_longitude(self):
        """Test longitude validation."""
        assert validate_type(45.5, Longitude) == 45.5
        assert validate_type(-179.9, Longitude) == -179.9

    def test_validate_longitude_out_of_range(self):
        """Test longitude validation with invalid values."""
        with pytest.raises(ValueError):
            validate_type(180, Longitude)
        with pytest.raises(ValueError):
            validate_type(-181, Longitude)

    def test_validate_elevation(self):
        """Test elevation validation."""
        assert validate_type(1000, Elevation) == 1000.0
        assert validate_type(0, Elevation) == 0.0

    def test_validate_elevation_out_of_range(self):
        """Test elevation validation with invalid values."""
        with pytest.raises(ValueError):
            validate_type(9000, Elevation)
        with pytest.raises(ValueError):
            validate_type(-500, Elevation)

    def test_validate_soda_timestep(self):
        """Test SodaTimeStep validation."""
        assert validate_type("PT01H", SodaTimeStep) == "PT01H"
        assert validate_type("PT15M", SodaTimeStep) == "PT15M"

    def test_validate_soda_stream(self):
        """Test SodaStream validation."""
        assert validate_type("mcclear", SodaStream) == "mcclear"
        assert validate_type("cams_radiation", SodaStream) == "cams_radiation"

    def test_validate_model(self):
        """Test Model validation."""
        assert validate_type("SPARTA", Model) == "SPARTA"
        assert validate_type("BIRD", Model) == "BIRD"

    def test_validate_none(self):
        """Test that None values pass through without validation."""
        assert validate_type(None, Latitude) is None
        assert validate_type(None, Model) is None

    def test_invalid_annotated_type(self):
        """Test error handling for non-annotated types."""
        with pytest.raises(TypeError, match="is not an Annotated type"):
            # Create a type that is not Annotated
            class FakeType:
                __value__ = float
            validate_type(42, FakeType())


class TestTypeAliases:
    """Test suite for type alias validation in real-world scenarios."""

    def test_latitude_spanish_cities(self):
        """Test validation with real Spanish city coordinates."""
        madrid_lat = validate_type(40.4168, Latitude)
        assert madrid_lat == 40.4168

        barcelona_lat = validate_type(41.3851, Latitude)
        assert barcelona_lat == 41.3851

    def test_longitude_spanish_cities(self):
        """Test validation with real Spanish city coordinates."""
        madrid_lon = validate_type(-3.7038, Longitude)
        assert madrid_lon == -3.7038

        barcelona_lon = validate_type(2.1734, Longitude)
        assert barcelona_lon == 2.1734

    def test_model_case_insensitive(self):
        """Test model name validation is case-insensitive."""
        assert validate_type("sparta", Model) == "SPARTA"
        assert validate_type("bird", Model) == "BIRD"
