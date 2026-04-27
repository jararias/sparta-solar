"""Pytest configuration and shared fixtures.

This module contains pytest fixtures that can be used across all test modules.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def sample_times():
    """Fixture that provides sample datetime array."""
    return pd.date_range("2020-01-01", periods=10, freq="D")


@pytest.fixture
def sample_coordinates():
    """Fixture that provides sample lat/lon coordinates."""
    return {
        "latitude": [36.7, 40.4, 28.5],  # Málaga, Madrid, Canarias
        "longitude": [-4.4, -3.7, -16.3],
    }


@pytest.fixture
def sample_atmosphere_data():
    """Fixture that provides a sample xarray Dataset with atmospheric data."""
    lat = np.linspace(25, 45, 20)
    lon = np.linspace(-20, 5, 25)
    time = pd.date_range("2020-01-01", periods=5, freq="D")
    
    ds = xr.Dataset(
        {
            "pressure": (["time", "lat", "lon"], np.random.uniform(950, 1050, (5, 20, 25))),
            "albedo": (["time", "lat", "lon"], np.random.uniform(0.1, 0.3, (5, 20, 25))),
            "pwater": (["time", "lat", "lon"], np.random.uniform(0.5, 3.0, (5, 20, 25))),
            "ozone": (["time", "lat", "lon"], np.random.uniform(0.25, 0.35, (5, 20, 25))),
            "beta": (["time", "lat", "lon"], np.random.uniform(0.05, 0.2, (5, 20, 25))),
            "alpha": (["time", "lat", "lon"], np.random.uniform(1.0, 1.5, (5, 20, 25))),
            "ssa": (["time", "lat", "lon"], np.random.uniform(0.85, 0.95, (5, 20, 25))),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "time": time,
        }
    )
    return ds


@pytest.fixture
def temp_config_dir(tmp_path):
    """Fixture that provides a temporary directory for config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sparta_input_params():
    """Fixture that provides typical input parameters for SPARTA model."""
    return {
        "cosz": 0.7,
        "pressure": 1013.25,
        "albedo": 0.2,
        "pwater": 1.5,
        "ozone": 0.3,
        "beta": 0.1,
        "alpha": 1.3,
        "ssa": 0.92,
        "asy": 0.65,
        "ecf": 1.0,
    }
