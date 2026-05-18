"""Unit tests for the BIRD clear-sky model.

Tests cover:
- Output interface (keys, types, shapes)
- Physical consistency (positive values, energy balance, nighttime masking)
- Parameter sensitivity (expected directional effects)
- Edge cases (cosz=0, NaN inputs, extreme values)
- Vectorization (array inputs, broadcasting)
"""
import numpy as np
import pytest

from spartasolar.modlib import BIRD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def typical_params():
    """Typical mid-latitude, midday atmospheric parameters."""
    return dict(
        cosz=0.7,
        pressure=1013.25,
        albedo=0.2,
        pwater=1.4,
        ozone=0.3,
        beta=0.1,
        alpha=1.3,
        ssa=0.92,
        asy=0.65,
        ecf=1.0,
    )


# ---------------------------------------------------------------------------
# Interface & output structure
# ---------------------------------------------------------------------------

class TestBIRDInterface:
    def test_returns_dict(self, typical_params):
        result = BIRD(**typical_params)
        assert isinstance(result, dict)

    def test_expected_keys(self, typical_params):
        result = BIRD(**typical_params)
        assert set(result.keys()) >= {"dni", "dhi", "dif", "ghi"}

    def test_no_csi_key(self, typical_params):
        """BIRD does not compute CSI, unlike SPARTA."""
        result = BIRD(**typical_params)
        assert "csi" not in result

    def test_output_types(self, typical_params):
        result = BIRD(**typical_params)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert isinstance(result[key], (np.ndarray, float, np.floating)), (
                f"{key} is not a numeric type: {type(result[key])}"
            )

    def test_scalar_input_produces_scalar_array(self, typical_params):
        result = BIRD(**typical_params)
        for key in ("dni", "dhi", "dif", "ghi"):
            # Scalar inputs may produce float or 0-d/1-element array
            val = result[key]
            arr = np.asarray(val)
            assert arr.size == 1, f"{key}: expected scalar, got shape {arr.shape}"

    def test_array_input_preserves_shape(self, typical_params):
        params = dict(typical_params)
        cosz_arr = np.array([0.3, 0.5, 0.7, 0.9])
        params["cosz"] = cosz_arr
        result = BIRD(**params)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert result[key].shape == cosz_arr.shape, f"{key} shape mismatch"


# ---------------------------------------------------------------------------
# Physical behaviour
# ---------------------------------------------------------------------------

class TestBIRDPhysics:
    def test_daytime_irradiance_positive(self, typical_params):
        result = BIRD(**typical_params)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert float(result[key]) > 0, f"{key} should be positive during daytime"

    def test_ghi_equals_dhi_plus_dif(self, typical_params):
        result = BIRD(**typical_params)
        np.testing.assert_allclose(
            result["ghi"], result["dhi"] + result["dif"], rtol=1e-5,
            err_msg="GHI must equal DHI + DIF"
        )

    def test_dhi_equals_dni_times_cosz(self, typical_params):
        """DHI = DNI × cos(SZA) by definition."""
        result = BIRD(**typical_params)
        np.testing.assert_allclose(
            result["dhi"], result["dni"] * typical_params["cosz"], rtol=1e-5,
            err_msg="DHI must equal DNI × cosz"
        )

    def test_ghi_bounded_by_extraterrestrial(self, typical_params):
        """GHI cannot exceed top-of-atmosphere irradiance."""
        SC = 1353.0
        etr = SC * typical_params["ecf"] * typical_params["cosz"]
        result = BIRD(**typical_params)
        assert float(result["ghi"]) <= etr * 1.05  # 5% tolerance

    def test_nighttime_returns_zero(self):
        """BIRD sets nighttime values to 0 (not NaN).
        
        Nighttime is defined as cosz <= cos(90.5°) ≈ -0.00873.
        """
        cosz_night = np.cos(np.radians(91.0))  # ≈ -0.01745, clearly below threshold
        result = BIRD(cosz=cosz_night)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert float(result[key]) == 0.0, f"{key} should be 0 at night"

    def test_nighttime_sza_gt_90_5(self):
        """Values for SZA > 90.5° (cosz < cos(90.5°)) must be zero."""
        cosz_night = np.cos(np.radians(91.0))
        result = BIRD(cosz=cosz_night)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert float(result[key]) == 0.0

    def test_mixed_day_night_array(self):
        """Array inputs with mixed daytime/nighttime.
        
        Nighttime threshold: cosz <= cos(90.5°) ≈ -0.00873.
        """
        cosz = np.array([0.7, 0.3, np.cos(np.radians(91.0)), np.cos(np.radians(100.0))])
        result = BIRD(cosz=cosz)
        for key in ("dni", "dhi", "dif", "ghi"):
            # last two are clearly nighttime → zero
            np.testing.assert_array_equal(result[key][-2:], 0.0)
            # first two are daytime → positive
            assert result[key][0] > 0
            assert result[key][1] > 0


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

class TestBIRDParameterEffects:
    def _get(self, key, **kwargs):
        return float(BIRD(**kwargs)[key])

    def test_higher_beta_reduces_dni(self, typical_params):
        """More aerosol → less DNI."""
        low = self._get("dni", **{**typical_params, "beta": 0.05})
        high = self._get("dni", **{**typical_params, "beta": 0.4})
        assert low > high

    def test_higher_pwater_reduces_dni(self, typical_params):
        low = self._get("dni", **{**typical_params, "pwater": 0.5})
        high = self._get("dni", **{**typical_params, "pwater": 4.0})
        assert low > high

    def test_lower_pressure_increases_dni(self, typical_params):
        """Lower pressure → less Rayleigh scattering → more DNI."""
        sea_level = self._get("dni", **{**typical_params, "pressure": 1013.25})
        altitude = self._get("dni", **{**typical_params, "pressure": 700.0})
        assert altitude > sea_level

    def test_higher_albedo_increases_ghi(self, typical_params):
        """Higher albedo → more multiple-reflection → more diffuse → more GHI."""
        low = self._get("ghi", **{**typical_params, "albedo": 0.05})
        high = self._get("ghi", **{**typical_params, "albedo": 0.8})
        assert high > low

    def test_larger_cosz_increases_ghi(self, typical_params):
        """Higher sun → more GHI."""
        low_sun = self._get("ghi", **{**typical_params, "cosz": 0.2})
        high_sun = self._get("ghi", **{**typical_params, "cosz": 0.95})
        assert high_sun > low_sun

    def test_ecf_scales_irradiance(self, typical_params):
        """Larger ecf (closer to Sun) → proportionally larger irradiance."""
        near = self._get("ghi", **{**typical_params, "ecf": 1.034})
        far = self._get("ghi", **{**typical_params, "ecf": 0.967})
        assert near > far


# ---------------------------------------------------------------------------
# Edge cases & robustness
# ---------------------------------------------------------------------------

class TestBIRDEdgeCases:
    def test_cosz_exactly_one(self):
        result = BIRD(cosz=1.0)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert np.isfinite(float(result[key]))

    def test_nan_cosz_propagates_nan(self):
        """NaN inputs should produce NaN outputs (not errors)."""
        result = BIRD(cosz=np.nan)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert np.isnan(float(result[key]))

    def test_extreme_high_beta(self, typical_params):
        """Very high aerosol load should still produce valid (near-zero) results."""
        result = BIRD(**{**typical_params, "beta": 5.0})
        for key in ("dni", "dhi", "dif", "ghi"):
            val = float(result[key])
            assert np.isfinite(val) and val >= 0.0

    def test_zero_ozone(self, typical_params):
        result = BIRD(**{**typical_params, "ozone": 0.0})
        for key in ("dni", "dhi", "dif", "ghi"):
            assert np.isfinite(float(result[key]))

    def test_all_nan_array(self):
        cosz = np.array([np.nan, np.nan])
        result = BIRD(cosz=cosz)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert np.all(np.isnan(result[key]))


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

class TestBIRDVectorisation:
    def test_1d_array(self, typical_params):
        params = dict(typical_params)
        params["cosz"] = np.linspace(0.1, 1.0, 20)
        result = BIRD(**params)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert result[key].shape == (20,)

    def test_2d_array(self, typical_params):
        params = dict(typical_params)
        params["cosz"] = np.full((4, 5), 0.7)
        result = BIRD(**params)
        for key in ("dni", "dhi", "dif", "ghi"):
            assert result[key].shape == (4, 5)

    def test_reproducibility(self, typical_params):
        r1 = BIRD(**typical_params)
        r2 = BIRD(**typical_params)
        for key in ("dni", "dhi", "dif", "ghi"):
            np.testing.assert_array_equal(r1[key], r2[key])
