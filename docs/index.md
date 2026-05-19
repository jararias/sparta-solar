<!-- <figure markdown="span">
![header](images/sunny-helmet-black-transparent.png){ width="250" align=left loading=lazy }
<strong>Batteries-included clear-sky solar radiation for solar resource assessment</strong>
</figure> -->

<!-- the caption element below is a workaround to center the image -->
![image title](images/sunny-helmet-black-transparent-recortada.png#only-light){ width="400" }
![image title](images/sunny-helmet-white-transparent-recortada.png#only-dark){ width="400" }
/// caption
///

# sparta-solar: A clear-sky solar radiation model, batteries included

<!-- <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
  <img src="images/sunny-helmet-black-transparent-recortada.png" width="250" alt="Logo" style="margin: 0;">
  <h1 style="margin: 0;">SPARTA Solar: A batteries-included clear-sky solar radiation model</h3>
</div> -->

![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![Tests](images/badges/tests.svg)](images/badges/tests.svg)
[![Coverage](images/badges/coverage.svg)](images/badges/coverage.svg)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**sparta-solar** is a Python library for computing **clear-sky solar irradiance** at the surface using the *Solar Parameterization of the Radiative Transfer of the Atmosphere* ([SPARTA](http://hdl.handle.net/10630/28011)) model. It integrates seamlessly with multiple atmospheric databases to supply the aerosol, water vapour, ozone, and pressure inputs that the radiative transfer model requires.

## Key features

#### Physics-based clear-sky irradiance

SPARTA is a high-accuracy **2-band broadband** clear-sky model that computes direct normal (DNI), global horizontal (GHI), diffuse horizontal (DIF) and, unique in its class, circumsolar irradiance (CSI). It is fully vectorized and handles nighttime
masking, surface albedo feedback and aerosol–Rayleigh coupling.

A legacy clear-sky model (Bird & Hulstrom, 1981) is also included for comparison and backward-compatibility studies.

#### Multiple atmospheric databases

**sparta-solar** ships out-of-the-box with ready-to-use connectors for several atmospheric datasets, covering different temporal resolutions, geographic extents, and access methods:

| Source | Module | Coverage | Resolution |
|---|---|---|---|
| NASA MERRA-2 reanalysis<br>long-term averages<br>shipped with the package | `merra2_lta` | 1999–2018 climatology<br>12 monthly values<br>global coverage<br>point and gridded | 0.5° × 0.625° |
| NASA MERRA-2 reanalysis<br>from a dedicated<br>[Hugging Face dataset](https://huggingface.co/datasets/josearuizarias/merra2-daily-clearsky) | `merra2_daily` | 1999–2018<br>daily time steps<br>global coverage<br>point and gridded | 0.5° × 0.625° |
| NASA MERRA-2 reanalysis<br>via [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog?hl=es-419)<br>(requires GEE account) | `merra2_gee` | 1980–present<br>hourly time steps<br>global coverage<br>point | 0.5° × 0.625° |
| CAMS radiation service<br>via [SODA](https://www.soda-pro.com/web-services/radiation/cams-mcclear) API<br>(requries SODA account) | `crs_soda` | 2004–present<br>hourly (averaged) time steps<br>global coverage<br>point | 0.75° × 0.75° |
| User-defined | `custom` | Any | Any |

#### CF-compliant xarray outputs

All outputs are returned as `xarray.Dataset` objects following
[Climate and Forecast (CF-1.11)](https://cfconventions.org/) metadata conventions — ready for analysis, visualization, or export to NetCDF/Zarr.

#### Flexible configuration

A TOML-based configuration system lets you set data directories, API credentials, and algorithm preferences once, globally, without cluttering your scripts.

---

## Quick examples

**Example 1 — Site time series with CAMS radiation service data from SODA**

```python
import pandas as pd
from spartasolar.atmosphere import crs_soda

times = pd.date_range("2020-06-15", periods=24, freq="h")
lat, lon, location = 36.72, -4.42, "Málaga"

# The first time that a calculation for this location and
# time period is requested, sparta-solar fetches the data
# from the SODA service and archive it locally for future
# calculations, if needed

atmos = crs_soda.at_site(
    times=times,
    latitude=lat,
    longitude=lon,
    site_names=location)

result = atmos.compute(model="SPARTA")

print(result)  # xarray.Dataset with ghi, dni, dif, csi, …
```

**Example 2 — Gridded irradiance map from MERRA-2 daily data**

```python
import numpy as np
import pandas as pd
from spartasolar.atmosphere import merra2_daily

lats = np.arange(36.0, 41.5, 0.5)
lons = np.arange(-9.0, -3.5, 0.5)
times = pd.date_range("2020-06-21 12:00", periods=1, freq="h")

# The first time that a calculation for this time period
# is requested, sparta-solar downloads the yearly dataset
# that covers this time period from hugging face hub and
# archive it locally for future calculations, if needed

atmos = merra2_daily.on_regular_grid(
  times=times,
  latitude=lats,
  longitude=lons)

result = atmos.compute(model="SPARTA")

result.ghi.isel(time=0).plot()  # solar noon GHI map over Iberian Peninsula
```

**Example 3 — Direct use of the SPARTA model**

```python
from spartasolar.modlib import SPARTA

out = SPARTA(
    cosz=0.866,       # cos(30°), ~30° solar zenith
    pressure=1013.25, # hPa
    pwater=2.0,       # cm
    ozone=0.3,        # atm-cm
    beta=0.05,        # low turbidity
    alpha=1.3,
    albedo=0.2,
)

print(f"GHI = {out['ghi']:.1f} W/m²,  DNI = {out['dni']:.1f} W/m²")
```

---

## Getting started

Ready to use sparta-solar? Start with the [Installation](installation.md) guide, then follow the [User Guide](user-guide.md) for a detailed walkthrough of every atmospheric data source.

---

## References

- **SPARTA model**: Ruiz-Arias, J. A. (2023). SPARTA: Solar parameterization for the radiative transfer of the cloudless atmosphere. *Renewable and Sustainable Energy Reviews*, 188, 113833. <https://doi.org/10.1016/j.rser.2023.113833>
- **MERRA-2 Reanalysis**: Gelaro, R., et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *J. Climate*, 30, 5419–5454. <https://doi.org/10.1175/JCLI-D-16-0758.1>
- **CRS/SODA**: CAMS solar radiation time-series data documentation [url](https://confluence.ecmwf.int/display/CKB/CAMS+solar+radiation+time-series%3A+data+documentation)
- **Bird model**: Bird, R. E. and Hulstrom, R. L. (1981). A Simplified Clear Atmosphere Radiative Transmittance Model. SERI/TR-642-1156.

## License

sparta-solar is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — free for non-commercial use with attribution.

## Contact

For any comment write me to jararias at uma dot es.

## Citation

If you use sparta-solar in your research, please cite:

```bibtex
@article{ruizarias2023sparta,
  author  = {Ruiz-Arias, Jose A.},
  title   = {{SPARTA}: Solar parameterization for the radiative transfer of the cloudless atmosphere},
  journal = {Renewable and Sustainable Energy Reviews},
  volume  = {188},
  pages   = {113833},
  year    = {2023},
  doi     = {10.1016/j.rser.2023.113833}
}
```
