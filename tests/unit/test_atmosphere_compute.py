"""Tests for BaseAtmosphere.compute() and the bundled/custom atmospheres (offline)."""

import numpy as np
import pandas as pd
import pytest

from spartasolar.atmoslib._base import validate_site_names
from spartasolar.atmoslib.helpers import pwater_in_kg_m2_to_cm, ozone_in_kg_m2_to_cm
from spartasolar.atmosphere import custom, merra2_cda, merra2_lta

TIMES = pd.date_range("2025-06-21", "2025-06-22", freq="1h")
LATS = [53.14, 36.72]
LONS = [8.21, -4.42]


class TestValidateSiteNames:

    def test_none_returns_range(self):
        assert validate_site_names(None, 3) == [0, 1, 2]

    @pytest.mark.parametrize("names", [[0, 1, 2], np.arange(3), range(3)])
    def test_integer_labels_roundtrip(self, names):
        assert validate_site_names(names, 3) == [0, 1, 2]

    def test_string_array(self):
        out = validate_site_names(np.array(["a", "b"]), 2)
        assert list(out) == ["a", "b"]

    def test_scalar_string(self):
        assert list(validate_site_names("a", 1)) == ["a"]

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            validate_site_names(["a", "b"], 3)


class TestMultiSiteCompute:

    def test_default_site_names(self):
        result = merra2_lta.at_sites(times=TIMES, latitude=LATS, longitude=LONS).compute()
        assert result.sizes == {"time": len(TIMES), "site": 2}
        assert list(result.site.values) == [0, 1]
        assert float(result.ghi.max()) > 0

    def test_custom_site_names(self):
        result = merra2_lta.at_sites(
            times=TIMES, latitude=LATS, longitude=LONS, site_names=["Oldenburg", "Malaga"]).compute()
        assert list(result.site.values) == ["Oldenburg", "Malaga"]

    def test_include_atmosphere(self):
        atmos = merra2_lta.at_sites(times=TIMES, latitude=LATS, longitude=LONS)
        result = atmos.compute(include_atmosphere=True)
        for var in ("ghi", "dni", "dif", "pwater", "ozone", "beta"):
            assert var in result.data_vars
        np.testing.assert_allclose(result.pwater.values, atmos.dataset.pwater.values)
        assert result.pwater.attrs["units"] == "kg m-2"

    def test_exclude_atmosphere_by_default(self):
        result = merra2_lta.at_sites(times=TIMES, latitude=LATS, longitude=LONS).compute()
        assert "pwater" not in result.data_vars


class TestCustomAtmosphereUnits:

    def _constituents_from_lta(self, n_sites):
        ds = merra2_lta.at_sites(times=TIMES, latitude=LATS[:n_sites], longitude=LONS[:n_sites]).dataset
        cons = {var: ds[var].values for var in ("pressure", "alpha", "beta", "ssa", "albedo")}
        cons["pwater"] = pwater_in_kg_m2_to_cm(ds["pwater"].values)  # cm
        cons["ozone"] = ozone_in_kg_m2_to_cm(ds["ozone"].values)  # atm-cm
        return cons

    def test_cm_and_atm_cm_reproduce_lta(self):
        expected = merra2_lta.at_sites(times=TIMES, latitude=LATS, longitude=LONS).compute()
        atmos = custom.at_sites(
            times=TIMES, latitude=LATS, longitude=LONS, constituents=self._constituents_from_lta(2))
        np.testing.assert_allclose(atmos.compute().ghi.values, expected.ghi.values, atol=1e-6)

    def test_stored_in_kg_m2(self):
        atmos = custom.at_sites(
            times=TIMES, latitude=36.72, longitude=-4.42,
            constituents={"pwater": np.full(len(TIMES), 2.0), "ozone": np.full(len(TIMES), 0.3)})
        np.testing.assert_allclose(atmos.dataset.pwater.values, 20.0)
        np.testing.assert_allclose(atmos.dataset.ozone.values, 0.3 * 2.1415e-2)

    def test_1d_input_single_site(self):
        cons = {k: np.ravel(v) for k, v in self._constituents_from_lta(1).items()}
        result = custom.at_sites(
            times=TIMES, latitude=LATS[0], longitude=LONS[0], constituents=cons, site_names="x").compute()
        assert result.sizes == {"time": len(TIMES), "site": 1}

    def test_regular_grid_units(self):
        shape = (len(TIMES), 2, 3)
        atmos = custom.on_regular_grid(
            times=TIMES, latitude=[36., 37.], longitude=[-5., -4., -3.],
            constituents={"pwater": np.full(shape, 1.5), "ozone": np.full(shape, 0.3)})
        np.testing.assert_allclose(atmos.dataset.pwater.values, 15.0)


class TestMERRA2CDA:

    def test_fixed_values_and_units(self):
        ds = merra2_cda.at_sites(times=TIMES, latitude=LATS, longitude=LONS).dataset
        np.testing.assert_allclose(pwater_in_kg_m2_to_cm(ds.pwater.values), 0.1)
        np.testing.assert_allclose(ds.beta.values, 0.01)
        assert ds.pwater.attrs["units"] == "kg m-2"

    def test_cleaner_than_lta(self):
        cda = merra2_cda.at_sites(times=TIMES, latitude=LATS, longitude=LONS).compute()
        lta = merra2_lta.at_sites(times=TIMES, latitude=LATS, longitude=LONS).compute()
        assert float(cda.dni.max()) > float(lta.dni.max())


class TestModelEdgeCases:

    @pytest.mark.parametrize("model", ["SPARTA", "BIRD"])
    def test_no_negative_irradiance_near_horizon(self, model):
        from spartasolar import modlib
        cosz = np.linspace(-0.0087, 0.01, 50)
        out = getattr(modlib, model)(cosz=cosz)
        for var in ("dni", "dhi", "dif", "ghi"):
            assert np.all(out[var] >= 0), var

    def test_bird_ozone_transmittance_lowers_dni(self):
        from spartasolar.modlib import BIRD
        # Bird & Hulstrom: To = 1 - 0.1611 uo (1+139.48 uo)^-0.3035 - 0.002715 uo / (1+0.044 uo+0.0003 uo^2)
        uo = 0.3
        to = 1 - 0.1611*uo/(1+139.48*uo)**0.3035 - 0.002715*uo/(1+0.044*uo+0.0003*uo**2)
        dni_o3 = BIRD(cosz=1., ozone=uo, pressure=1013.25)["dni"]
        dni_no_o3 = BIRD(cosz=1., ozone=0., pressure=1013.25)["dni"]
        np.testing.assert_allclose(dni_o3 / dni_no_o3, to, rtol=1e-6)

    def test_sparta_csi_param(self):
        from spartasolar.modlib import SPARTA
        assert float(SPARTA(cosz=0.8, beta=0.3, csi_param="Sparta")["csi"]) > 0
        assert float(SPARTA(cosz=0.8, beta=0.3, csi_param="none")["csi"]) == 0
        with pytest.raises(ValueError, match="csi_param"):
            SPARTA(cosz=0.8, csi_param="spartaa")


def test_database_path_follows_config_changes(tmp_path):
    from spartasolar import config
    from spartasolar.atmoslib.crs_sodaapi import CRSSODAAtmosphere
    previous = config.get_option("crs_soda.data_dir")
    try:
        config.set_option("crs_soda.data_dir", tmp_path)
        assert CRSSODAAtmosphere.database_path == tmp_path
    finally:
        if previous is not None:
            config.set_option("crs_soda.data_dir", previous)
