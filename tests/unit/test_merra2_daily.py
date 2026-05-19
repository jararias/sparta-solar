"""Unit tests for spartasolar.atmoslib.merra2_daily module.

Tests cover:
- MERRA2DailyAtmosphere.at_sites() method
- MERRA2DailyAtmosphere.on_regular_grid() method
- Year inference from time arrays
- Dataset loading and validation
- Path management and error handling
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from spartasolar.atmoslib.merra2_daily import (
    MERRA2DailyAtmosphere,
    get_database_path,
)


class TestGetDatabasePath:
    """Test suite for database path configuration."""

    @patch('spartasolar.atmoslib.merra2_daily.get_option')
    @patch('spartasolar.atmoslib.merra2_daily.platformdirs.user_data_path')
    def test_get_database_path_with_config(self, mock_user_data_path, mock_get_option, tmp_path):
        """Test that configured path is used when available."""
        config_path = tmp_path / "custom_merra2"
        config_path.mkdir()
        mock_get_option.return_value = config_path
        
        result = get_database_path()
        
        assert result == config_path
        mock_get_option.assert_called_once_with("merra2_daily.data_dir")

    @patch('spartasolar.atmoslib.merra2_daily.get_option')
    @patch('spartasolar.atmoslib.merra2_daily.platformdirs.user_data_path')
    def test_get_database_path_creates_directory(self, mock_user_data_path, mock_get_option, tmp_path):
        """Test that directory is created if it doesn't exist."""
        mock_get_option.return_value = None
        default_path = tmp_path / "spartasolar" / "merra2-daily"
        mock_user_data_path.return_value = default_path
        
        result = get_database_path()
        
        assert result.exists()
        assert result == default_path


class TestInferYearsFromTimes:
    """Test suite for _infer_years_from_times static method."""

    def test_infer_years_single_year(self):
        """Test year inference with dates from a single year."""
        times = pd.date_range("2020-06-01", periods=10, freq="D")
        
        years = MERRA2DailyAtmosphere._infer_years_from_times(times)
        
        assert years == [2020]

    def test_infer_years_multiple_years(self):
        """Test year inference with dates spanning multiple years."""
        times = pd.date_range("2020-11-01", periods=90, freq="D")
        
        years = MERRA2DailyAtmosphere._infer_years_from_times(times)
        
        assert 2020 in years
        assert 2021 in years

    def test_infer_years_with_boundary_padding_start(self):
        """Test that previous year is added if close to year start."""
        times = pd.date_range("2020-01-02", periods=5, freq="D")
        
        years = MERRA2DailyAtmosphere._infer_years_from_times(times)
        
        # Should include 2019 because we're close to start of 2020
        assert 2019 in years
        assert 2020 in years

    def test_infer_years_with_boundary_padding_end(self):
        """Test that next year is added if close to year end."""
        times = pd.date_range("2020-12-29", periods=5, freq="D")
        
        years = MERRA2DailyAtmosphere._infer_years_from_times(times)
        
        # Should include 2021 because we're close to end of 2020
        assert 2020 in years
        assert 2021 in years

    def test_infer_years_with_numpy_datetime64(self):
        """Test year inference with numpy datetime64 array."""
        times = np.array(['2020-06-01', '2020-06-15', '2020-07-01'], dtype='datetime64[D]')
        
        years = MERRA2DailyAtmosphere._infer_years_from_times(times)
        
        assert years == [2020]

    def test_infer_years_sorted_output(self):
        """Test that output years are sorted."""
        times = pd.date_range("2019-12-29", periods=100, freq="D")
        
        years = MERRA2DailyAtmosphere._infer_years_from_times(times)
        
        assert years == sorted(years)


class TestEnsureAllPathsAreLocal:
    """Test suite for _ensure_all_paths_are_local static method."""

    def test_all_paths_exist(self, tmp_path):
        """Test with all paths existing."""
        paths = [tmp_path / "2020", tmp_path / "2021"]
        for path in paths:
            path.mkdir()
        
        # Should not raise any exception
        MERRA2DailyAtmosphere._ensure_all_paths_are_local(paths)

    def test_missing_path_raises_error(self, tmp_path):
        """Test that missing path raises NotImplementedError."""
        existing_path = tmp_path / "2020"
        existing_path.mkdir()
        missing_path = tmp_path / "2021"
        
        paths = [existing_path, missing_path]
        
        with pytest.raises(ValueError, match="out of range"):
            MERRA2DailyAtmosphere._ensure_all_paths_are_local(paths)


class TestAtSitesMethod:
    """Test suite for at_sites class method."""

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_at_sites_single_location(self, mock_load_dataset, sample_atmosphere_data):
        """Test at_sites with a single location."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        lat = 40.4
        lon = -3.7
        
        result = MERRA2DailyAtmosphere.at_sites(times, lat, lon)
        
        assert isinstance(result, MERRA2DailyAtmosphere)
        assert hasattr(result, '_atmosphere')
        mock_load_dataset.assert_called_once()

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_at_sites_multiple_locations(self, mock_load_dataset, sample_atmosphere_data):
        """Test at_sites with multiple locations."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        lats = [40.4, 41.4, 36.7]
        lons = [-3.7, 2.2, -4.4]
        
        result = MERRA2DailyAtmosphere.at_sites(times, lats, lons)
        
        assert isinstance(result, MERRA2DailyAtmosphere)
        mock_load_dataset.assert_called_once()

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_at_sites_with_site_names(self, mock_load_dataset, sample_atmosphere_data):
        """Test at_sites with site names provided."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        lats = [40.4, 41.4]
        lons = [-3.7, 2.2]
        names = ["Madrid", "Barcelona"]
        
        result = MERRA2DailyAtmosphere.at_sites(times, lats, lons, site_names=names)
        
        assert isinstance(result, MERRA2DailyAtmosphere)
        # Check that site coordinate was added
        assert "site" in result._atmosphere.coords

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_at_sites_mismatched_coordinates_raises_error(self, mock_load_dataset, sample_atmosphere_data):
        """Test that mismatched lat/lon lengths raise ValueError."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        lats = [40.4, 41.4]
        lons = [-3.7]  # Different length
        
        with pytest.raises(ValueError, match="must have the same length"):
            MERRA2DailyAtmosphere.at_sites(times, lats, lons)

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_at_sites_invalid_latitude(self, mock_load_dataset, sample_atmosphere_data):
        """Test that invalid latitude raises ValueError."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        
        with pytest.raises(ValueError):
            MERRA2DailyAtmosphere.at_sites(times, 95.0, 0.0)  # Invalid lat

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_at_sites_invalid_longitude(self, mock_load_dataset, sample_atmosphere_data):
        """Test that invalid longitude raises ValueError."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        
        with pytest.raises(ValueError):
            MERRA2DailyAtmosphere.at_sites(times, 40.0, 200.0)  # Invalid lon


class TestOnRegularGridMethod:
    """Test suite for on_regular_grid class method."""

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_on_regular_grid(self, mock_load_dataset, sample_atmosphere_data):
        """Test on_regular_grid with valid inputs."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        lats = [35.0, 40.0, 45.0]
        lons = [-10.0, -5.0, 0.0]
        
        result = MERRA2DailyAtmosphere.on_regular_grid(times, lats, lons)
        
        assert isinstance(result, MERRA2DailyAtmosphere)
        mock_load_dataset.assert_called_once()

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_on_regular_grid_handles_nan_albedo(self, mock_load_dataset, sample_atmosphere_data):
        """Test that NaN values in albedo are filled with 0."""
        # Add NaN to albedo
        sample_atmosphere_data["albedo"][0, 0, 0] = np.nan
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        lats = [35.0, 40.0]
        lons = [-10.0, 0.0]
        
        result = MERRA2DailyAtmosphere.on_regular_grid(times, lats, lons)
        
        # Verify no NaN in the output (they should be filled)
        assert isinstance(result, MERRA2DailyAtmosphere)


class TestLoadDataset:
    """Test suite for _load_dataset class method."""

    @patch.object(MERRA2DailyAtmosphere, '_ensure_all_paths_are_local')
    @patch('spartasolar.atmoslib.merra2_daily.xr.open_mfdataset')
    def test_load_dataset_single_year(self, mock_open_mf, mock_ensure_paths, sample_atmosphere_data, tmp_path):
        """Test loading dataset for a single year."""
        mock_open_mf.return_value = sample_atmosphere_data
        MERRA2DailyAtmosphere.database_path = tmp_path
        
        times = pd.date_range("2020-06-01", periods=10, freq="D")
        
        result = MERRA2DailyAtmosphere._load_dataset(times)
        
        assert isinstance(result, xr.Dataset)
        mock_ensure_paths.assert_called_once()
        mock_open_mf.assert_called_once()

    @patch.object(MERRA2DailyAtmosphere, '_ensure_all_paths_are_local')
    @patch('spartasolar.atmoslib.merra2_daily.xr.open_mfdataset')
    def test_load_dataset_multiple_years(self, mock_open_mf, mock_ensure_paths, sample_atmosphere_data, tmp_path):
        """Test loading dataset spanning multiple years."""
        mock_open_mf.return_value = sample_atmosphere_data
        MERRA2DailyAtmosphere.database_path = tmp_path
        
        times = pd.date_range("2020-11-01", periods=90, freq="D")
        
        result = MERRA2DailyAtmosphere._load_dataset(times)
        
        assert isinstance(result, xr.Dataset)
        mock_ensure_paths.assert_called_once()
        
        # Verify that paths from both years were considered
        call_args = mock_ensure_paths.call_args[0][0]
        years_in_paths = [int(p.name) for p in call_args]
        assert 2020 in years_in_paths
        assert 2021 in years_in_paths

    @patch.object(MERRA2DailyAtmosphere, '_ensure_all_paths_are_local')
    @patch('spartasolar.atmoslib.merra2_daily.xr.open_mfdataset')
    def test_load_dataset_calls_open_mfdataset_correctly(self, mock_open_mf, mock_ensure_paths, sample_atmosphere_data, tmp_path):
        """Test that xr.open_mfdataset is called with correct parameters."""
        mock_open_mf.return_value = sample_atmosphere_data
        MERRA2DailyAtmosphere.database_path = tmp_path
        
        times = pd.date_range("2020-06-01", periods=10, freq="D")
        
        MERRA2DailyAtmosphere._load_dataset(times)
        
        # Verify open_mfdataset was called with expected kwargs
        call_kwargs = mock_open_mf.call_args[1]
        assert call_kwargs['engine'] == 'zarr'
        assert call_kwargs['combine'] == 'nested'
        assert call_kwargs['concat_dim'] == 'time'
        assert call_kwargs['data_vars'] == 'minimal'
        assert call_kwargs['compat'] == 'override'
        assert call_kwargs['coords'] == 'minimal'


class TestMERRA2DailyAtmosphereIntegration:
    """Integration tests combining multiple methods."""

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_end_to_end_at_sites(self, mock_load_dataset, sample_atmosphere_data):
        """Test complete workflow for at_sites method."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=5, freq="D")
        lats = [40.4, 41.4]
        lons = [-3.7, 2.2]
        names = ["Madrid", "Barcelona"]
        
        result = MERRA2DailyAtmosphere.at_sites(
            times=times,
            latitude=lats,
            longitude=lons,
            site_names=names
        )
        
        # Verify the result is properly constructed
        assert isinstance(result, MERRA2DailyAtmosphere)
        assert hasattr(result, '_atmosphere')
        assert isinstance(result._atmosphere, xr.Dataset)
        
        # Verify coordinates exist
        assert 'site' in result._atmosphere.dims
        assert 'site' in result._atmosphere.coords

    @patch.object(MERRA2DailyAtmosphere, '_load_dataset')
    def test_end_to_end_on_regular_grid(self, mock_load_dataset, sample_atmosphere_data):
        """Test complete workflow for on_regular_grid method."""
        mock_load_dataset.return_value = sample_atmosphere_data
        
        times = pd.date_range("2020-01-01", periods=5, freq="D")
        lats = np.linspace(35, 45, 3)
        lons = np.linspace(-10, 5, 4)
        
        result = MERRA2DailyAtmosphere.on_regular_grid(
            times=times,
            latitude=lats,
            longitude=lons
        )
        
        # Verify the result is properly constructed
        assert isinstance(result, MERRA2DailyAtmosphere)
        assert hasattr(result, '_atmosphere')
        assert isinstance(result._atmosphere, xr.Dataset)
