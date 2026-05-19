# User Guide

This guide explains how **sparta-solar** is organized and walks you through every atmospheric data source with practical examples.

## Core concepts

### The two-step workflow

**sparta-solar** separates **atmospheric data retrieval** from **irradiance
computation**. Every workflow follows the same two-step pattern:

```python
# Step 1 — load atmospheric constituents into an atmosphere instance, e.g., merra2_daily
atmos = merra2_daily.at_sites(times=times, latitude=lats, longitude=lons)

# Step 2 — compute clear-sky irradiance from that atmosphere instance
result = atmos.compute(model="SPARTA")
```

`atmos` is a `BaseAtmosphere` instance whose `.dataset` property holds an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) with the atmospheric variables (the constituents of the clear-sky atmosphere) needed by the model:

| Variable | Description | Units |
|---|---|---|
| `pressure` | Surface pressure | Pa |
| `pwater` | Total-column water vapor amount,<br>or precipitable water | kg m-2 |
| `ozone` | Total column ozone content | kg m-2 |
| `beta` | Ångström turbidity coefficient | – |
| `alpha` | Ångström wavelength exponent | – |
| `ssa` | Aerosol single-scattering albedo | – |
| `albedo` | Surface albedo | – |

**The constituents are spatially and temporally interpolated** from each gridded data source to the requested timestamps and locations.

After `.compute()`, the returned dataset contains:

| Variable | Description | Units |
|---|---|---|
| `ghi` | Clear-sky global horizontal irradiance | W m⁻² |
| `dni` | Clear-sky direct normal irradiance | W m⁻² |
| `dif` | Clear-sky diffuse horizontal irradiance | W m⁻² |
| `dhi` | Clear-sky direct horizontal irradiance | W m⁻² |
| `csi` | Clear-sky circumsolar irradiance | W m⁻² |
| `cosz` | Cosine of solar zenith angle | – |

### Two spatial layouts

All atmosphere classes support one or two data layouts intended for sites and regular grids (see Table below). The layout for sites is provided by all atmospheres (via *`at_sites()`* in gridded atmospheres and *`at_site()`* in point-only atmospheres). The layout for regular grids is provided only by the gridded atmospheres (via *`on_regular_grid()`*).

| Atmosphere | *`at_site()`* | *`at_sites()`* | *`on_regular_grid()`* |
|---         |:---:|:---:|:---:|
|`merra2_lta`  |     | ✔️  | ✔️  |
|`merra2_daily`|     | ✔️  | ✔️  |
|`merra2_gee`  | ✔️  |     |     |
|`crs_soda`    | ✔️  |     |     |

**Example 1: `at_sites()` — one or several fixed locations sharing a common time grid.
Output dimensions: `(time, site)`**

```python
atmos = merra2_daily.at_sites(
    times=times,
    latitude=[36.72, 40.42],
    longitude=[-4.42, -3.70],
    site_names=["Málaga", "Madrid"],
)
print(atmos.dataset.dims)  # {'time': N, 'site': 2}
```

**Example 2: `on_regular_grid()` — a rectangular latitude × longitude grid.
Output dimensions: `(time, lat, lon)`**

```python
atmos = merra2_daily.on_regular_grid(
    times=times,
    latitude=np.arange(36, 41, 0.5),
    longitude=np.arange(-9, -3, 0.5),
)
print(atmos.dataset.dims)  # {'time': N, 'lat': 10, 'lon': 12}
```

### CF-compliant metadata

All datasets carry
[CF-1.11](https://cfconventions.org/) global attributes (title, institution,
source, history, references, featureType) and per-variable attributes
(`standard_name`, `long_name`, `units`). They can be written directly to NetCDF:

```python
result.to_netcdf("clear_sky_irradiance.nc")
```

### Choosing a clear-sky model

The `compute()` method accepts a `model` keyword:

```python
result_sparta = atmos.compute(model="SPARTA")   # default
result_bird   = atmos.compute(model="Bird")     # Bird & Hulstrom, 1981
```

Additional model-specific parameters can be passed via `model_kwargs`:

```python
result = atmos.compute(
    model="SPARTA",
    model_kwargs={"csi_hfov": 2.9},  # alternate pyrheliometer FOV
)
```

---

## MERRA-2 daily reanalysis

The `merra2_daily` atmosphere automagically downloads NASA MERRA-2 daily files from [Hugging Face Hub](https://huggingface.co/) and caches them locally. **No credentials are needed**. It is the recommended source for applications that require gridded outputs or access to many locations.

**Coverage**: Global, 1999–2018  
**Spatial resolution**: 0.5° latitude × 0.625° longitude
**Temporal resolution**: Daily (averaged from hourly)

### Configuration

Set the local cache directory once via the configuration system. For instance:

```python
from spartasolar import config
config.set_option("merra2_daily.data_dir", "/data/merra2-daily")
```

If not set, **sparta-solar** uses a platform-appropriate user data directory[^1].

[^1]: In a Linux system, `~/.local/share/sparta-solar/merra2-daily`. See `user_data_dir` at [platformdirs](https://platformdirs.readthedocs.io/en/latest/platforms.html#user-data-dir) for further details.

### Usage

For one location:

```python
import pandas as pd
from spartasolar.atmosphere import merra2_daily

times = pd.date_range("2020-06-01", "2020-06-30", freq="h")

atmos = merra2_daily.at_sites(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    site_names="Málaga",
)

# Inspect the atmospheric dataset
print(atmos.dataset)

# Compute irradiance
result = atmos.compute(model="SPARTA")
print(result)
```

For multiple sites:

```python
atmos = merra2_daily.at_sites(
    times=times,
    latitude=[36.72, 40.42, 41.38],
    longitude=[-4.42, -3.70,  2.17],
    site_names=["Málaga", "Madrid", "Barcelona"],
)

result = atmos.compute()
print(result.ghi.sel(site="Madrid"))
```

For a regular grid of locations:

```python
import numpy as np

lats = np.arange(36.0, 44.0, 0.5)
lons = np.arange(-9.0, 4.0, 0.5)
times = pd.date_range("2020-06-21 12:00", periods=1, freq="h")

atmos = merra2_daily.on_regular_grid(times=times, latitude=lats, longitude=lons)
result = atmos.compute()

# Plot summer-solstice noon GHI
result.ghi.isel(time=0).plot(cmap="YlOrRd")
```

---

## MERRA-2 long-term averages

The `merra2_lta` atmosphere source uses a pre-computed climatology (1999–2018 monthly
averages) bundled within the package. **No downloads or API credentials are needed.**
It is well-suited for long-term assessments or when recent daily data is not required.

**Coverage**: Global
**Spatial resolution**: 0.5° latitude × 0.625° longitude
**Temporal basis**: Monthly climatology (12 monthly values per grid-cell)  

### Usage

The API is identical to `merra2_daily`:

```python
from spartasolar.atmosphere import merra2_lta

atmos = merra2_lta.at_sites(
    times=pd.date_range("2020-01-01", "2020-12-31", freq="h"),
    latitude=36.72,
    longitude=-4.42,
    site_names="Málaga",
)

result = atmos.compute()
```

!!! note
    Because this source uses monthly climatology, each hour within a given
    calendar month will have the same atmospheric constituents (averaged across
    all years in the 1999–2018 baseline). Intra-monthly variability is not captured.

---

## MERRA-2 via Google Earth Engine

The `merra2_gee` source retrieves MERRA-2 hourly data through the
[Google Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install). It is useful when you want natively represent the atmosphere with hourly resolution, as simulated by MERRA-2.

**Coverage**: Global, 1980–present  
**Requires**: Google Earth Engine account and project

### Setup

1. Create a Google Earth Engine account at <https://earthengine.google.com/>.
2. Create or select a Cloud project with the Earth Engine API enabled.
3. Authenticate once from the command line:

    ```bash
    earthengine authenticate
    ```

4. Tell **sparta-solar** which project to use:

    ```python
    from spartasolar import config
    config.set_option("merra2_gee.project", "your-gee-project-id")
    config.set_option("merra2_gee.data_dir", "/data/merra2_gee")  # optional custom local cache
    ```

### Usage

```python
import pandas as pd
from spartasolar.atmosphere import merra2_gee

times = pd.date_range("2020-06-15", periods=24, freq="h")

atmos = merra2_gee.at_site(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    site_name="Málaga",
)

result = atmos.compute()
```

!!! note
    Data retrieved from GEE is cached locally in `data_dir` or, if not set, in a platform-appropriate user data directory[^2]. Subsequent calls for the same location and time range read from the cache without hitting the GEE API.

[^2]: In a Linux system, `~/.local/share/sparta-solar/merra2_gee`. See `user_data_dir` at [platformdirs](https://platformdirs.readthedocs.io/en/latest/platforms.html#user-data-dir) for further details.

---

## Copernicus CRS via SODA API

The `crs_soda` source retrieves atmospheric optical properties from the
[Copernicus Radiation Service (CRS)](https://www.soda-pro.com/web-services/radiation/cams-mcclear)
REST API. The data are derived from the CAMS (Copernicus Atmosphere Monitoring
Service) reanalysis.

**Coverage**: Global  
**Available period**: 2004–present  
**Temporal resolution**: Hourly  

### Setup

Register at <https://www.soda-pro.com/> (free) and configure your email:

```python
from spartasolar import config
config.set_option("crs_soda.user_email", "you@example.com")
config.set_option("crs_soda.data_dir", "/data/crs_soda")  # optional custom local data dir
```

### Usage

```python
import pandas as pd
from spartasolar.atmosphere import crs_soda

times = pd.date_range("2020-06-01", "2020-06-30", freq="h")

atmos = crs_soda.at_site(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    site_name="Málaga",
)

result = atmos.compute()
```

!!! note
    The SODA API imposes rate limits. Retrieved data are cached locally so that
    repeated calls for the same period do not generate additional API requests.

---

## Custom atmosphere

The `custom` source lets you supply your own atmospheric constituents.
This is useful for sensitivity studies, model inter-comparisons, or when you
already have the atmospheric inputs from another source.

### Required variables

At minimum you must provide `pressure`, `pwater`, `ozone`, `beta`, `alpha`, and
`albedo`. Optionally you can also provide `ssa` (single-scattering albedo) and
`asy` (asymmetry parameter).

### Usage

```python
import numpy as np
import pandas as pd
from spartasolar.atmosphere import custom

n = 24  # number of time steps
times = pd.date_range("2020-06-21", periods=n, freq="h")

constituents = {
    "pressure": np.full(n, 101325.0),   # Pa
    "pwater":   np.linspace(1.0, 3.0, n),  # cm
    "ozone":    np.full(n, 0.3),         # atm-cm
    "beta":     np.full(n, 0.05),
    "alpha":    np.full(n, 1.3),
    "albedo":   np.full(n, 0.15),
}

atmos = custom.at_sites(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    constituents=constituents,
    site_names="Málaga",
)

result = atmos.compute()
```

---

## Configuration system

**sparta-solar** stores user settings in a TOML file under the platform's user
configuration directory (e.g. `~/.config/sparta-solar/config.toml` on Linux).

### View current configuration

```python
from spartasolar import config
config.show_config()
```

### Read an option

```python
data_dir = config.get_option("merra2_daily.data_dir")
```

### Set an option

```python
config.set_option("merra2_daily.data_dir", "/my/data")
config.set_option("crs_soda.user_email", "you@example.com")
```

### Reset to defaults

```python
config.reset_config_file()
```

---

## Working with xarray outputs

All outputs are `xarray.Dataset` or `xarray.DataArray` objects.

```python
result = atmos.compute()

# Select by coordinate value
result.ghi.sel(site="Málaga")
result.ghi.sel(time="2020-06-21 12:00")

# Select by index
result.ghi.isel(time=0)

# Aggregate
result.ghi.mean(dim="time")
result.ghi.resample(time="D").max()

# Export
result.to_netcdf("output.nc")
result.to_zarr("output.zarr")
result.ghi.to_pandas()
```

---

## Next steps

- Browse the [Quick Reference](quick-reference.md) for a concise per-source
  cheat-sheet.
- Explore the [API Reference](api.md) for full parameter documentation.
