# Testing Guide for SPARTA-Solar

This document describes the test suite for the SPARTA-Solar library, how to run tests, and how to interpret the results.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Coverage Reports](#coverage-reports)
- [Continuous Integration](#continuous-integration)

## Overview

The SPARTA-Solar test suite uses **pytest** as the testing framework, with **pytest-cov** for coverage reporting and **pytest-mock** for mocking capabilities. The tests are organized into unit tests and integration tests to ensure code quality and correctness.

### Test Coverage

The test suite currently covers:

- ✅ **Validation module** (`validation.py`) - Type validators and annotated types
- ✅ **Configuration module** (`config.py`) - Configuration file management
- ✅ **SPARTA model** (`modlib/sparta.py`) - Solar radiation model
- ✅ **MERRA2 Daily Atmosphere** (`atmoslib/merra2_daily.py`) - Atmospheric data handling

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests
│   ├── test_validation.py      # Tests for validation module
│   ├── test_config.py          # Tests for configuration module
│   ├── test_sparta.py          # Tests for SPARTA model
│   └── test_merra2_daily.py    # Tests for MERRA2 atmosphere
├── integration/                # Integration tests (future)
└── fixtures/                   # Test data files (future)
```

### Shared Fixtures

The `conftest.py` file contains shared pytest fixtures available to all tests:

- `sample_times` - Sample datetime arrays
- `sample_coordinates` - Sample lat/lon coordinates (Spanish cities)
- `sample_atmosphere_data` - Mock xarray Dataset with atmospheric data
- `temp_config_dir` - Temporary directory for config files
- `sparta_input_params` - Typical SPARTA model parameters

## Running Tests

### Prerequisites

Install development dependencies:

```bash
uv sync --dev
```

Or if using pip:

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests with verbose output
uv run pytest -v

# Run all tests with coverage
uv run pytest --cov=spartasolar --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Run only validation tests
uv run pytest tests/unit/test_validation.py -v

# Run only SPARTA model tests
uv run pytest tests/unit/test_sparta.py -v

# Run only config tests
uv run pytest tests/unit/test_config.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
uv run pytest tests/unit/test_validation.py::TestValidaRange -v

# Run a specific test method
uv run pytest tests/unit/test_sparta.py::TestSPARTABasics::test_sparta_returns_dict -v
```

### Run Tests by Markers

```bash
# Run only unit tests
uv run pytest -m unit

# Run only slow tests
uv run pytest -m slow

# Run only integration tests
uv run pytest -m integration
```

### Useful Pytest Options

```bash
# Show print statements
uv run pytest -v -s

# Stop at first failure
uv run pytest -x

# Run last failed tests
uv run pytest --lf

# Show local variables in tracebacks
uv run pytest -l

# Increase verbosity
uv run pytest -vv
```

## Writing Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure

```python
import pytest
from spartasolar.validation import ValidaRange


class TestValidaRange:
    """Test suite for ValidaRange validator."""

    def test_within_range(self):
        """Test that values within range pass validation."""
        validator = ValidaRange(ge=0, le=100)
        assert validator.validate(50) == 50.0

    def test_outside_range_raises_error(self):
        """Test that values outside range raise ValueError."""
        validator = ValidaRange(ge=0, le=100)
        with pytest.raises(ValueError):
            validator.validate(101)
```

### Using Fixtures

```python
def test_sparta_with_fixture(sparta_input_params):
    """Test using the sparta_input_params fixture."""
    from spartasolar.modlib.sparta import SPARTA
    
    result = SPARTA(**sparta_input_params)
    assert result['dni'] > 0
```

### Mocking External Dependencies

```python
from unittest.mock import patch

@patch('spartasolar.atmoslib.merra2_daily.xr.open_mfdataset')
def test_load_dataset(mock_open_mf, sample_atmosphere_data):
    """Test dataset loading with mocked xarray."""
    mock_open_mf.return_value = sample_atmosphere_data
    
    result = MERRA2DailyAtmosphere._load_dataset(times)
    assert isinstance(result, xr.Dataset)
```

## Coverage Reports

### Viewing Coverage in Terminal

```bash
uv run pytest --cov=spartasolar --cov-report=term-missing
```

This shows:
- Percentage of code covered
- Line numbers that are not covered

### Generating HTML Coverage Report

```bash
uv run pytest --cov=spartasolar --cov-report=html
```

Open `htmlcov/index.html` in your browser to view detailed coverage reports with highlighted source code.

### Coverage Configuration

Coverage settings are configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src/spartasolar"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/site-packages/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

## Test Categories

### Unit Tests

Unit tests verify individual components in isolation:

- **Validation tests**: Type validators, range checks, pattern matching
- **Config tests**: Configuration file reading/writing
- **SPARTA tests**: Model physics, parameter sensitivity, edge cases
- **MERRA2 tests**: Data loading, interpolation, coordinate handling

### Integration Tests (Future)

Integration tests will verify that components work together correctly:

- End-to-end atmospheric data retrieval
- Complete radiation modeling workflows
- Real dataset processing

## Continuous Integration

Tests should be run automatically on:

- Every commit to feature branches
- Pull requests to main branch
- Before creating releases

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev
      - name: Run tests
        run: uv run pytest --cov=spartasolar
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you've installed the package in development mode:

```bash
uv sync --dev
# or
pip install -e .
```

### Fixture Not Found

Ensure fixtures are defined in `conftest.py` or imported correctly:

```python
# In conftest.py
@pytest.fixture
def my_fixture():
    return "value"

# In test file - fixture is automatically available
def test_something(my_fixture):
    assert my_fixture == "value"
```

### Mocking Issues

When mocking, use the full import path where the object is used, not where it's defined:

```python
# If merra2_daily.py imports: from ..config import get_option
# Mock it where it's used:
@patch('spartasolar.atmoslib.merra2_daily.get_option')
```

## Best Practices

1. **One assertion per test** when possible
2. **Use descriptive test names** that explain what is being tested
3. **Add docstrings** to test methods explaining the test purpose
4. **Mock external dependencies** (file I/O, network calls, database)
5. **Use fixtures** for common test data
6. **Test edge cases** and error conditions
7. **Keep tests fast** - slow tests should be marked with `@pytest.mark.slow`
8. **Aim for high coverage** but prioritize testing critical paths

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
