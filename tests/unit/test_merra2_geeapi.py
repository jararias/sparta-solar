"""Unit and integration tests for MERRA2GEEAtmosphere.

Tests use a small parquet file in tests/data/merra2_gee/ so no GEE
credentials or network access are required.  The test data covers
lat=37.5°N, lon=3.5°W for the year 2015.

The `merra2_gee.data_dir` config option is used to redirect the class
to the local test data directory.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Test data constants – must match the files in tests/data/merra2_gee/
# ---------------------------------------------------------------------------
TEST_LAT = 37.5
TEST_LON = -3.5
TEST_SITE = "Jayena"
# Mid-March 2015: safely away from year boundary (no padding year needed)
TEST_TIMES = pd.date_range("2015-03-05 00:30", "2015-03-10 23:30", freq="h")

DATA_DIR = Path(__file__).parent.parent / "data" / "merra2_gee"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gee_atmosphere():
    """MERRA2GEEAtmosphere loaded from local test data."""
    from spartasolar import config
    from spartasolar.atmoslib.merra2_geeapi import MERRA2GEEAtmosphere

    # Point to test data via config option; at_site() resolves the path from config
    config.set_option("merra2_gee.data_dir", DATA_DIR)
    return MERRA2GEEAtmosphere.at_site(
        times=TEST_TIMES,
        latitude=TEST_LAT,
        longitude=TEST_LON,
        site_name=TEST_SITE,
    )


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

class TestGEEDatasetStructure:
    def test_returns_atmosphere_instance(self, gee_atmosphere):
        from spartasolar.atmoslib.merra2_geeapi import MERRA2GEEAtmosphere
        assert isinstance(gee_atmosphere, MERRA2GEEAtmosphere)

    def test_dataset_is_xarray(self, gee_atmosphere):
        assert isinstance(gee_atmosphere.dataset, xr.Dataset)

    def test_expected_variables(self, gee_atmosphere):
        for var in ("pressure", "ozone", "pwater", "beta", "alpha", "ssa", "albedo"):
            assert var in gee_atmosphere.dataset, f"missing variable '{var}'"

    def test_time_dimension(self, gee_atmosphere):
        assert gee_atmosphere.dataset.sizes["time"] == len(TEST_TIMES)

    def test_site_dimension(self, gee_atmosphere):
        assert gee_atmosphere.dataset.sizes["site"] == 1

    def test_site_name(self, gee_atmosphere):
        sites = gee_atmosphere.dataset.coords["site"].values
        assert sites[0] == TEST_SITE

    def test_lat_lon_coordinates(self, gee_atmosphere):
        np.testing.assert_allclose(
            gee_atmosphere.dataset.coords["lat"].values, [TEST_LAT], atol=1e-6)
        np.testing.assert_allclose(
            gee_atmosphere.dataset.coords["lon"].values, [TEST_LON], atol=1e-6)

    def test_time_coordinate_matches_request(self, gee_atmosphere):
        ds_times = pd.DatetimeIndex(gee_atmosphere.dataset.coords["time"].values)
        pd.testing.assert_index_equal(ds_times, TEST_TIMES, check_names=False)


# ---------------------------------------------------------------------------
# Physical plausibility of atmospheric variables
# ---------------------------------------------------------------------------

class TestGEEAtmosphericVariables:
    def _values(self, gee_atmosphere, var):
        return gee_atmosphere.dataset[var].values.ravel()

    def test_pressure_range(self, gee_atmosphere):
        """Surface pressure in Pa: 50000–110000 Pa."""
        p = self._values(gee_atmosphere, "pressure")
        assert np.all(np.isfinite(p))
        assert np.all((p >= 50000) & (p <= 110000))

    def test_pwater_non_negative(self, gee_atmosphere):
        pw = self._values(gee_atmosphere, "pwater")
        assert np.all(np.isfinite(pw) & (pw >= 0))

    def test_ozone_positive(self, gee_atmosphere):
        oz = self._values(gee_atmosphere, "ozone")
        assert np.all(np.isfinite(oz) & (oz > 0))

    def test_beta_non_negative(self, gee_atmosphere):
        beta = self._values(gee_atmosphere, "beta")
        assert np.all(np.isfinite(beta) & (beta >= 0))

    def test_alpha_positive(self, gee_atmosphere):
        alpha = self._values(gee_atmosphere, "alpha")
        assert np.all(np.isfinite(alpha) & (alpha > 0))

    def test_ssa_bounded(self, gee_atmosphere):
        ssa = self._values(gee_atmosphere, "ssa")
        assert np.all(np.isfinite(ssa) & (ssa > 0) & (ssa <= 1))

    def test_albedo_bounded(self, gee_atmosphere):
        alb = self._values(gee_atmosphere, "albedo")
        assert np.all(np.isfinite(alb) & (alb >= 0) & (alb <= 1))


# ---------------------------------------------------------------------------
# Clearsky model output via compute()
# ---------------------------------------------------------------------------

class TestGEEComputeSPARTA:
    @pytest.fixture(autouse=True)
    def _run(self, gee_atmosphere):
        self.result = gee_atmosphere.compute(model="SPARTA")

    def test_returns_dataset(self):
        assert isinstance(self.result, xr.Dataset)

    def test_has_irradiance_variables(self):
        for var in ("dni", "dhi", "dif", "ghi"):
            assert var in self.result

    def test_ghi_non_negative_daytime(self):
        ghi = self.result["ghi"].values
        finite = ghi[np.isfinite(ghi)]
        assert np.all(finite >= 0)

    def test_ghi_energy_balance(self):
        ghi = self.result["ghi"].values
        dhi = self.result["dhi"].values
        dif = self.result["dif"].values
        mask = np.isfinite(ghi)
        np.testing.assert_allclose(ghi[mask], (dhi + dif)[mask], rtol=1e-4)

    def test_dimensions(self):
        assert self.result.sizes["time"] == len(TEST_TIMES)
        assert self.result.sizes["site"] == 1


class TestGEEComputeBIRD:
    @pytest.fixture(autouse=True)
    def _run(self, gee_atmosphere):
        self.result = gee_atmosphere.compute(model="BIRD")

    def test_returns_dataset(self):
        assert isinstance(self.result, xr.Dataset)

    def test_has_irradiance_variables(self):
        for var in ("dni", "dhi", "dif", "ghi"):
            assert var in self.result

    def test_ghi_non_negative(self):
        ghi = self.result["ghi"].values
        finite = ghi[np.isfinite(ghi) & (ghi > 0)]
        assert np.all(finite >= 0)

    def test_ghi_energy_balance(self):
        ghi = self.result["ghi"].values
        dhi = self.result["dhi"].values
        dif = self.result["dif"].values
        mask = np.isfinite(ghi) & (ghi > 0)
        np.testing.assert_allclose(ghi[mask], (dhi + dif)[mask], rtol=1e-4)


# ---------------------------------------------------------------------------
# SPARTA vs BIRD consistency on GEE atmosphere
# ---------------------------------------------------------------------------

class TestGEESPARTAvsBIRD:
    @pytest.fixture(autouse=True)
    def _run(self, gee_atmosphere):
        self.sparta = gee_atmosphere.compute(model="SPARTA")
        self.bird = gee_atmosphere.compute(model="BIRD")

    def _daytime_mask(self):
        s = self.sparta["ghi"].values
        b = self.bird["ghi"].values
        return np.isfinite(s) & (s > 0) & np.isfinite(b) & (b > 0)

    @pytest.mark.parametrize("var,tol", [
        ("ghi", 0.20),
        ("dhi", 0.20),
        ("dif", 0.30),
        ("dni", 0.25),
    ])
    def test_mean_relative_difference(self, var, tol):
        mask = self._daytime_mask()
        s = self.sparta[var].values[mask]
        b = self.bird[var].values[mask]
        mrd = np.mean(np.abs(s - b) / (0.5 * (s + b)))
        assert mrd < tol, f"{var}: mean rel. diff {mrd:.2%} > {tol:.0%}"

    def test_sparta_ghi_bird_ghi_ratio(self):
        mask = self._daytime_mask()
        s = self.sparta["ghi"].values[mask]
        b = self.bird["ghi"].values[mask]
        ratio = np.mean(s) / np.mean(b)
        assert 0.85 <= ratio <= 1.15, f"SPARTA/BIRD GHI ratio: {ratio:.3f}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestGEEReproducibility:
    def test_dataset_reproducible(self, gee_atmosphere):
        r1 = gee_atmosphere.compute(model="SPARTA")
        r2 = gee_atmosphere.compute(model="SPARTA")
        np.testing.assert_array_equal(r1["ghi"].values, r2["ghi"].values)
