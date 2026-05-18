![header](images/headerfig.png)

![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![Tests](../badges/tests.svg)](../badges/tests.svg)
[![Coverage](../badges/coverage.svg)](../badges/coverage.svg)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**sparta-solar** is a Python library for computing **clear-sky solar irradiance** at the surface
using the SPARTA (Solar PArameterization of the Radiative Transfer of the Atmosphere) model.
It integrates seamlessly with multiple atmospheric databases to supply the aerosol, water vapour,
ozone, and pressure inputs that the radiative transfer model requires.

## Key features

#### Physics-based clear-sky irradiance

SPARTA is a high-accuracy **2-band broadband** clear-sky model that computes direct
normal (DNI), global horizontal (GHI), diffuse horizontal (DHI), and circumsolar
irradiance (CSI) simultaneously. It is fully vectorized and handles nighttime
masking, surface albedo feedback, and aerosol–Rayleigh coupling.

A legacy clear-sky model (Bird & Hulstrom, 1981) is also included for comparison
and backward-compatibility studies.

#### Multiple atmospheric databases

sparta-solar ships with ready-to-use connectors for several atmospheric datasets,
covering different temporal resolutions, geographic extents, and access methods:

| Source | Module | Coverage | Resolution |
|---|---|---|---|
| NASA MERRA-2 daily reanalysis | `merra2_daily` | 1980–present, global | 0.5° × 0.625° |
| MERRA-2 long-term averages | `merra2_lta` | 1999–2018 climatology, global | 0.5° × 0.625° |
| MERRA-2 via Google Earth Engine | `merra2_gee` | 1980–present, global | 0.5° × 0.625° |
| Copernicus CRS via SODA API | `crs_soda` | 2004–present, Europe/Africa | 1-min (resampled) |
| User-defined | `custom` | Any | Any |

#### CF-compliant xarray outputs

All outputs are returned as `xarray.Dataset` objects following
[Climate and Forecast (CF-1.11)](https://cfconventions.org/) metadata conventions —
ready for analysis, visualization, or export to NetCDF/Zarr.

#### Flexible configuration

A TOML-based configuration system lets you set data directories, API credentials,
and algorithm preferences once, globally, without cluttering your scripts.

---

## Quick examples

**Example 1 — Site time series with MERRA-2 daily data**

```python
import pandas as pd
from spartasolar.atmosphere import merra2_daily

times = pd.date_range("2020-06-15", periods=24, freq="h")

atm = merra2_daily.at_sites(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    site_names="Málaga",
)

result = atm.compute(model="SPARTA")

print(result)  # xarray.Dataset with ghi, dni, dif, csi, …
```

**Example 2 — Gridded irradiance map**

```python
import numpy as np
import pandas as pd
from spartasolar.atmosphere import merra2_daily

lats = np.arange(36.0, 41.5, 0.5)
lons = np.arange(-9.0, -3.5, 0.5)
times = pd.date_range("2020-06-21 12:00", periods=1, freq="h")

atm = merra2_daily.on_regular_grid(times=times, latitude=lats, longitude=lons)
result = atm.compute(model="SPARTA")

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

Ready to use sparta-solar? Start with the [Installation](installation.md) guide,
then follow the [User Guide](user-guide.md) for a detailed walkthrough of every
atmospheric data source.

---

## References

- **SPARTA model**: Ruiz-Arias, J. A. (2023). SPARTA: Solar parameterization for the
  radiative transfer of the cloudless atmosphere. *Renewable and Sustainable Energy
  Reviews*, 188, 113833. <https://doi.org/10.1016/j.rser.2023.113833>
- **MERRA-2**: Gelaro, R., et al. (2017). The Modern-Era Retrospective Analysis for
  Research and Applications, Version 2 (MERRA-2). *J. Climate*, 30, 5419–5454.
  <https://doi.org/10.1175/JCLI-D-16-0758.1>
- **CRS/SODA**: Gschwind, B., et al. (2019). Improving the McClear model estimating the
  downwelling solar radiation at ground level in cloud-free conditions. *Met. Apps.*,
  26, 571–576. <https://doi.org/10.1002/met.1774>
- **Bird model**: Bird, R. E. and Hulstrom, R. L. (1981). A Simplified Clear Atmosphere
  Radiative Transmittance Model. SERI/TR-642-1156.

## License

sparta-solar is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) —
free for non-commercial use with attribution.

## Citation

If you use sparta-solar in your research, please cite:

```bibtex
@article{ruizarias2023sparta,
  author  = {Ruiz-Arias, Jose A.},
  title   = {{SPARTA}: Solar parameterization for the radiative transfer
             of the cloudless atmosphere},
  journal = {Renewable and Sustainable Energy Reviews},
  volume  = {188},
  pages   = {113833},
  year    = {2023},
  doi     = {10.1016/j.rser.2023.113833}
}
```
