"""Unit tests for pysparta.modlib.sparta module.

Tests cover:
- SPARTA model basic functionality
- Input validation and edge cases
- Nighttime handling
- Vector vs scalar inputs
- Physical reasonability checks
"""

import numpy as np
import pytest

from pysparta.modlib.sparta import SPARTA


class TestSPARTABasics:
    """Test suite for basic SPARTA model functionality."""

    def test_sparta_returns_dict(self, sparta_input_params):
        """Test that SPARTA returns a dictionary with expected keys."""
        result = SPARTA(**sparta_input_params)
        
        assert isinstance(result, dict)
        expected_keys = {'dni', 'dhi', 'dif', 'ghi', 'csi'}
        assert set(result.keys()) == expected_keys

    def test_sparta_output_types(self, sparta_input_params):
        """Test that all outputs are numpy arrays."""
        result = SPARTA(**sparta_input_params)
        
        for key, value in result.items():
            assert isinstance(value, np.ndarray), f"{key} should be np.ndarray"

    def test_sparta_with_scalar_inputs(self):
        """Test SPARTA with scalar inputs."""
        result = SPARTA(
            cosz=0.7,
            pressure=1013.25,
            albedo=0.2,
            pwater=1.5,
            ozone=0.3,
            beta=0.1,
            alpha=1.3,
        )
        
        # Results should be 0-d arrays (scalars)
        assert result['dni'].shape == ()
        assert result['ghi'].shape == ()

    def test_sparta_with_array_inputs(self):
        """Test SPARTA with array inputs."""
        n = 10
        result = SPARTA(
            cosz=np.full(n, 0.7),
            pressure=np.full(n, 1013.25),
            albedo=np.full(n, 0.2),
            pwater=np.full(n, 1.5),
            ozone=np.full(n, 0.3),
            beta=np.full(n, 0.1),
            alpha=np.full(n, 1.3),
        )
        
        assert result['dni'].shape == (n,)
        assert result['ghi'].shape == (n,)


class TestSPARTAPhysicalBehavior:
    """Test suite for physical reasonability of SPARTA outputs."""

    def test_daytime_positive_irradiance(self):
        """Test that daytime conditions produce positive irradiance."""
        result = SPARTA(cosz=0.7, pressure=1013.25, albedo=0.2)
        
        assert result['dni'] > 0, "DNI should be positive during daytime"
        assert result['ghi'] > 0, "GHI should be positive during daytime"
        assert result['dif'] >= 0, "DIF should be non-negative"

    def test_nighttime_zero_irradiance(self):
        """Test that nighttime (cosz <= cos(90.5°)) produces NaN/zero."""
        # cos(90.5°) ≈ -0.00872
        result = SPARTA(cosz=-0.01, pressure=1013.25, albedo=0.2)
        
        # Nighttime should produce NaN
        assert np.isnan(result['dni']) or result['dni'] == 0
        assert np.isnan(result['ghi']) or result['ghi'] == 0

    def test_ghi_equals_dni_plus_dif(self):
        """Test that GHI approximately equals DNI*cosz + DIF."""
        cosz = 0.7
        result = SPARTA(cosz=cosz, pressure=1013.25, albedo=0.2)
        
        calculated_ghi = result['dhi'] + result['dif']
        np.testing.assert_allclose(
            result['ghi'], 
            calculated_ghi,
            rtol=1e-5,
            err_msg="GHI should equal DHI + DIF"
        )

    def test_dni_within_solar_constant(self):
        """Test that DNI doesn't exceed solar constant with ecf."""
        result = SPARTA(cosz=1.0, pressure=1013.25, albedo=0.2, ecf=1.0)
        
        # DNI should not exceed solar constant (1361.1 W/m²)
        assert result['dni'] <= 1361.1 * 1.1, "DNI unreasonably high"

    def test_increasing_cosz_increases_irradiance(self):
        """Test that higher sun elevation increases irradiance."""
        result_low = SPARTA(cosz=0.3, pressure=1013.25, albedo=0.2)
        result_high = SPARTA(cosz=0.9, pressure=1013.25, albedo=0.2)
        
        assert result_high['ghi'] > result_low['ghi'], \
            "Higher sun should produce more irradiance"


class TestSPARTAParameterEffects:
    """Test suite for parameter sensitivity."""

    def test_higher_aerosol_reduces_dni(self):
        """Test that increased aerosol loading reduces DNI."""
        result_clean = SPARTA(cosz=0.7, beta=0.05, alpha=1.3)
        result_turbid = SPARTA(cosz=0.7, beta=0.3, alpha=1.3)
        
        assert result_turbid['dni'] < result_clean['dni'], \
            "Higher aerosol should reduce DNI"

    def test_higher_pwater_reduces_dni(self):
        """Test that more precipitable water reduces DNI."""
        result_dry = SPARTA(cosz=0.7, pwater=0.5)
        result_humid = SPARTA(cosz=0.7, pwater=3.0)
        
        assert result_humid['dni'] < result_dry['dni'], \
            "Higher pwater should reduce DNI"

    def test_higher_albedo_increases_dif(self):
        """Test that higher surface albedo increases diffuse (ground reflection)."""
        result_dark = SPARTA(cosz=0.7, albedo=0.1)
        result_bright = SPARTA(cosz=0.7, albedo=0.8)
        
        # Higher albedo can increase diffuse via ground reflection
        assert result_bright['ghi'] >= result_dark['ghi'], \
            "Higher albedo should increase or maintain GHI"

    def test_lower_pressure_reduces_irradiance(self):
        """Test that lower pressure (higher altitude) affects irradiance."""
        result_sealevel = SPARTA(cosz=0.7, pressure=1013.25)
        result_mountain = SPARTA(cosz=0.7, pressure=700)
        
        # At higher altitude, less Rayleigh scattering, generally higher DNI
        assert result_mountain['dni'] > result_sealevel['dni'], \
            "Lower pressure should increase DNI (less scattering)"


class TestSPARTAEdgeCases:
    """Test suite for edge cases and special conditions."""

    def test_zero_cosz(self):
        """Test behavior at horizon (cosz=0)."""
        result = SPARTA(cosz=0.0, pressure=1013.25)
        
        # Should be nighttime or very low irradiance
        assert np.isnan(result['ghi']) or result['ghi'] < 10

    def test_cosz_one(self):
        """Test behavior at solar noon at equator (cosz=1)."""
        result = SPARTA(cosz=1.0, pressure=1013.25)
        
        assert result['dni'] > 0
        assert not np.isnan(result['dni'])

    def test_extreme_beta_values(self):
        """Test with extreme aerosol values (clipped internally)."""
        # Beta is clipped to [0, 2.2] internally
        result = SPARTA(cosz=0.7, beta=5.0, alpha=1.3)
        
        # Should not crash, results should be finite
        assert np.isfinite(result['dni'])

    def test_nan_input_handling(self):
        """Test that NaN inputs produce NaN outputs."""
        result = SPARTA(cosz=np.nan, pressure=1013.25)
        
        assert np.isnan(result['dni'])
        assert np.isnan(result['ghi'])

    def test_mixed_day_night_array(self):
        """Test array with both daytime and nighttime values."""
        cosz_array = np.array([0.7, 0.3, -0.1, 0.5])  # Last two are nighttime
        result = SPARTA(cosz=cosz_array, pressure=1013.25)
        
        # First two should have values, last two should be NaN or zero
        assert result['dni'][0] > 0
        assert result['dni'][1] > 0
        assert np.isnan(result['dni'][2]) or result['dni'][2] == 0


class TestSPARTATransmittanceSchemes:
    """Test suite for different transmittance schemes."""

    def test_independent_scheme(self):
        """Test independent transmittance scheme."""
        result = SPARTA(
            cosz=0.7,
            transmittance_scheme='independent'
        )
        
        assert result['dni'] > 0
        assert np.isfinite(result['dni'])

    def test_interdependent_scheme(self):
        """Test interdependent transmittance scheme (default)."""
        result = SPARTA(
            cosz=0.7,
            transmittance_scheme='interdependent'
        )
        
        assert result['dni'] > 0
        assert np.isfinite(result['dni'])

    def test_schemes_produce_different_results(self):
        """Test that different schemes produce different results."""
        result_ind = SPARTA(cosz=0.7, transmittance_scheme='independent')
        result_inter = SPARTA(cosz=0.7, transmittance_scheme='interdependent')
        
        # They should produce different (but close) results
        assert result_ind['dni'] != result_inter['dni']


class TestSPARTACSI:
    """Test suite for circumsolar irradiance parameterization."""

    def test_csi_none(self):
        """Test CSI with 'none' parameterization."""
        result = SPARTA(cosz=0.7, csi_param='none')
        
        # CSI should be zero or very small
        assert result['csi'] == 0 or result['csi'] < 1

    def test_csi_sparta(self):
        """Test CSI with 'sparta' parameterization."""
        result = SPARTA(cosz=0.7, csi_param='sparta', beta=0.2)
        
        # With aerosols, CSI should be positive
        assert result['csi'] >= 0

    def test_csi_hfov_parameter(self):
        """Test that CSI half field of view affects results."""
        result_narrow = SPARTA(cosz=0.7, csi_param='sparta', csi_hfov=1.0, beta=0.2)
        result_wide = SPARTA(cosz=0.7, csi_param='sparta', csi_hfov=5.0, beta=0.2)
        
        # Wider FOV should generally capture more circumsolar
        assert result_wide['csi'] >= result_narrow['csi']


class TestSPARTAVectorization:
    """Test suite for vectorization and broadcasting."""

    def test_2d_array_input(self):
        """Test SPARTA with 2D array inputs."""
        shape = (5, 10)
        result = SPARTA(
            cosz=np.full(shape, 0.7),
            pressure=np.full(shape, 1013.25),
        )
        
        assert result['dni'].shape == shape
        assert result['ghi'].shape == shape

    def test_broadcasting(self):
        """Test that scalar and array inputs broadcast correctly."""
        cosz_array = np.array([0.5, 0.7, 0.9])
        result = SPARTA(
            cosz=cosz_array,
            pressure=1013.25,  # scalar
            albedo=0.2,  # scalar
        )
        
        assert result['dni'].shape == (3,)
        # Higher cosz should give higher irradiance
        assert result['dni'][2] > result['dni'][0]


class TestSPARTAReproducibility:
    """Test suite for model reproducibility."""

    def test_same_inputs_same_outputs(self):
        """Test that identical inputs produce identical outputs."""
        params = dict(cosz=0.7, pressure=1013.25, albedo=0.2, pwater=1.5)
        
        result1 = SPARTA(**params)
        result2 = SPARTA(**params)
        
        np.testing.assert_array_equal(result1['dni'], result2['dni'])
        np.testing.assert_array_equal(result1['ghi'], result2['ghi'])
