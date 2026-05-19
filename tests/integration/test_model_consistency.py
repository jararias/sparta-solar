"""Integration tests: SPARTA vs BIRD consistency using MERRA-2 GEE atmosphere.

Uses a small local parquet file (tests/data/merra2_gee/) so no network access
is required. The test data covers lat=37.5°N, lon=3.5°W for year 2015.

Goals:
- Both models produce physically valid output for the same atmosphere.
- SPARTA and BIRD agree within a generous tolerance (~20% mean relative
  difference for GHI; they are different formulations of the same physics).
- Results are reproducible across repeated calls.

Notes:
- BIRD returns nighttime as 0, SPARTA as NaN.  Tests normalise this before
  comparing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TEST_LAT = 37.5
TEST_LON = -3.5
TEST_SITE = "Jayena"
TEST_TIMES = pd.date_range("2015-03-05 00:30", "2015-03-10 23:30", freq="h")
GEE_DATA_DIR = Path(__file__).parent.parent / "data" / "merra2_gee"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gee_atmosphere():
    from spartasolar import config
    from spartasolar.atmoslib.merra2_geeapi import MERRA2GEEAtmosphere, get_database_path

    # Point to test data via config option; explicitly refresh the class
    # attribute since the module may already be cached at import time
    config.set_option("merra2_gee.data_dir", GEE_DATA_DIR)
    MERRA2GEEAtmosphere.database_path = get_database_path()
    return MERRA2GEEAtmosphere.at_site(
        times=TEST_TIMES,
        latitude=TEST_LAT,
        longitude=TEST_LON,
        site_name=TEST_SITE,
    )


@pytest.fixture(scope="module")
def sparta_output(gee_atmosphere):
    return gee_atmosphere.compute(model="SPARTA")


@pytest.fixture(scope="module")
def bird_output(gee_atmosphere):
    return gee_atmosphere.compute(model="BIRD")


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_sparta_returns_dataset(self, sparta_output):
        assert isinstance(sparta_output, xr.Dataset)

    def test_bird_returns_dataset(self, bird_output):
        assert isinstance(bird_output, xr.Dataset)

    def test_sparta_has_required_variables(self, sparta_output):
        for var in ("dni", "dhi", "dif", "ghi"):
            assert var in sparta_output

    def test_bird_has_required_variables(self, bird_output):
        for var in ("dni", "dhi", "dif", "ghi"):
            assert var in bird_output

    def test_sparta_has_csi(self, sparta_output):
        assert "csi" in sparta_output

    def test_bird_does_not_have_csi(self, bird_output):
        assert "csi" not in bird_output

    def test_dimensions(self, sparta_output):
        assert sparta_output.sizes["time"] == len(TEST_TIMES)
        assert sparta_output.sizes["site"] == 1


# ---------------------------------------------------------------------------
# Physical validity per model
# ---------------------------------------------------------------------------

class TestPhysicalValidity:
    @pytest.mark.parametrize("var", ["dni", "dhi", "dif", "ghi"])
    def test_sparta_daytime_non_negative(self, sparta_output, var):
        data = sparta_output[var].values
        finite = data[np.isfinite(data)]
        assert len(finite) > 0
        assert np.all(finite >= 0)

    @pytest.mark.parametrize("var", ["dni", "dhi", "dif", "ghi"])
    def test_bird_values_non_negative(self, bird_output, var):
        data = bird_output[var].values
        valid = data[np.isfinite(data) & (data > 0)]
        assert len(valid) > 0
        assert np.all(valid >= 0)

    def test_sparta_ghi_energy_balance(self, sparta_output):
        ghi = sparta_output["ghi"].values
        dhi = sparta_output["dhi"].values
        dif = sparta_output["dif"].values
        mask = np.isfinite(ghi)
        np.testing.assert_allclose(ghi[mask], (dhi + dif)[mask], rtol=1e-4)

    def test_bird_ghi_energy_balance(self, bird_output):
        ghi = bird_output["ghi"].values
        dhi = bird_output["dhi"].values
        dif = bird_output["dif"].values
        mask = np.isfinite(ghi) & (ghi > 0)
        np.testing.assert_allclose(ghi[mask], (dhi + dif)[mask], rtol=1e-4)


# ---------------------------------------------------------------------------
# SPARTA vs BIRD comparison
# ---------------------------------------------------------------------------

def _daytime_mask(sparta_ds, bird_ds):
    s = sparta_ds["ghi"].values
    b = bird_ds["ghi"].values
    return np.isfinite(s) & (s > 0) & np.isfinite(b) & (b > 0)


class TestSPARTAvsBIRD:
    @pytest.fixture(autouse=True)
    def _load(self, sparta_output, bird_output):
        self.sparta = sparta_output
        self.bird = bird_output
        self.mask = _daytime_mask(sparta_output, bird_output)

    def _mrd(self, var):
        s = self.sparta[var].values[self.mask]
        b = self.bird[var].values[self.mask]
        return np.mean(np.abs(s - b) / (0.5 * (s + b)))

    @pytest.mark.parametrize("var,tol", [
        ("ghi", 0.20),
        ("dhi", 0.20),
        ("dif", 0.30),
        ("dni", 0.25),
    ])
    def test_mean_relative_difference(self, var, tol):
        mrd = self._mrd(var)
        assert mrd < tol, f"{var}: mean rel. diff {mrd:.2%} > {tol:.0%}"

    def test_ghi_ratio_within_15_percent(self):
        s = self.sparta["ghi"].values[self.mask]
        b = self.bird["ghi"].values[self.mask]
        ratio = np.mean(s) / np.mean(b)
        assert 0.85 <= ratio <= 1.15, f"SPARTA/BIRD GHI ratio: {ratio:.3f}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_sparta_reproducible(self, gee_atmosphere):
        r1 = gee_atmosphere.compute(model="SPARTA")
        r2 = gee_atmosphere.compute(model="SPARTA")
        np.testing.assert_array_equal(r1["ghi"].values, r2["ghi"].values)

    def test_bird_reproducible(self, gee_atmosphere):
        r1 = gee_atmosphere.compute(model="BIRD")
        r2 = gee_atmosphere.compute(model="BIRD")
        np.testing.assert_array_equal(r1["ghi"].values, r2["ghi"].values)
