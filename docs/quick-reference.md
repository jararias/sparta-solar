# Quick Reference

A concise cheat-sheet organized by atmospheric data source.

---

## MERRA-2 daily reanalysis (`merra2_daily`)

**Module**: `spartasolar.atmosphere.merra2_daily`  
**Class**: `MERRA2DailyAtmosphere`  
**Coverage**: Global, 1980–present | **Res.**: 0.5° × 0.625° | **Access**: Hugging Face Hub (cached)

### Configuration keys

| Key | Description | Default |
|---|---|---|
| `merra2_daily.data_dir` | Local cache directory | Platform user-cache |
| `merra2_daily.project` | HuggingFace dataset identifier | (built-in) |

### at_sites — fixed locations

```python
import pandas as pd
from spartasolar.atmosphere import merra2_daily

times = pd.date_range("2020-06-01", "2020-06-30", freq="h")

atm = merra2_daily.at_sites(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    site_names="Málaga",   # optional; use a list for multiple sites
)

result = atm.compute(model="SPARTA")   # or model="Bird"
```

### on_regular_grid — lat × lon grid

```python
import numpy as np

lats = np.arange(36.0, 44.0, 0.5)
lons = np.arange(-9.0, 4.0, 0.5)
times = pd.date_range("2020-06-21 12:00", periods=1, freq="h")

atm = merra2_daily.on_regular_grid(times=times, latitude=lats, longitude=lons)
result = atm.compute()

result.dni.isel(time=0).plot()
```

### Output dimensions

| Method | Dimensions |
|---|---|
| `at_sites()` | `(time, site)` |
| `on_regular_grid()` | `(time, lat, lon)` |

---

## MERRA-2 long-term averages (`merra2_lta`)

**Module**: `spartasolar.atmosphere.merra2_lta`  
**Class**: `MERRA2LTAAtmosphere`  
**Coverage**: Global | **Basis**: 1999–2018 monthly climatology | **Access**: Bundled (no download)

### Usage

```python
from spartasolar.atmosphere import merra2_lta

atm = merra2_lta.at_sites(
    times=pd.date_range("2020-01-01", "2020-12-31", freq="h"),
    latitude=36.72,
    longitude=-4.42,
    site_names="Málaga",
)

result = atm.compute()
```

!!! tip
    `merra2_lta` requires no configuration and no internet connection —
    ideal for quick climatological estimates or offline environments.

---

## MERRA-2 via Google Earth Engine (`merra2_gee`)

**Module**: `spartasolar.atmosphere.merra2_gee`  
**Class**: `MERRA2GEEAtmosphere`  
**Coverage**: Global, 1980–present | **Access**: GEE API (requires account)

### One-time setup

```bash
pip install earthengine-api
earthengine authenticate
```

```python
from spartasolar import config
config.set_option("merra2_gee.project", "my-gee-project")
config.set_option("merra2_gee.data_dir", "/data/merra2_gee")
```

### Usage

```python
import pandas as pd
from spartasolar.atmosphere import merra2_gee

times = pd.date_range("2020-06-15", periods=24, freq="h")

atm = merra2_gee.at_site(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    site_name="Málaga",
)

result = atm.compute()
```

### Configuration keys

| Key | Description |
|---|---|
| `merra2_gee.project` | GEE Cloud project ID |
| `merra2_gee.data_dir` | Local cache directory |

---

## Copernicus CRS / SODA API (`crs_soda`)

**Module**: `spartasolar.atmosphere.crs_soda`  
**Class**: `CRSSODAAtmosphere`  
**Coverage**: Europe, Africa, and surroundings | **Period**: 2004–present | **Access**: REST API (free registration)

### One-time setup

Register at <https://www.soda-pro.com/> (free), then:

```python
from spartasolar import config
config.set_option("crs_soda.user_email", "you@example.com")
config.set_option("crs_soda.data_dir", "/data/crs_soda")
```

### Usage

```python
import pandas as pd
from spartasolar.atmosphere import crs_soda

times = pd.date_range("2020-06-01", "2020-06-30", freq="h")

atm = crs_soda.at_site(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    site_name="Málaga",
)

result = atm.compute()
```

### Configuration keys

| Key | Description |
|---|---|
| `crs_soda.user_email` | Registered SODA email address |
| `crs_soda.data_dir` | Local cache directory |

---

## Custom atmosphere (`custom`)

**Module**: `spartasolar.atmosphere.custom`  
**Class**: `CustomAtmosphere`  
**Coverage**: User-defined | **Access**: No external data needed

### Required constituent keys

`pressure` (Pa), `pwater` (cm), `ozone` (atm-cm), `beta`, `alpha`, `albedo`.
Optional: `ssa`, `asy`.

### Usage

```python
import numpy as np
import pandas as pd
from spartasolar.atmosphere import custom

n = 24
times = pd.date_range("2020-06-21", periods=n, freq="h")

atm = custom.at_sites(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    constituents={
        "pressure": np.full(n, 101325.0),
        "pwater":   np.linspace(1.0, 3.0, n),
        "ozone":    np.full(n, 0.3),
        "beta":     np.full(n, 0.05),
        "alpha":    np.full(n, 1.3),
        "albedo":   np.full(n, 0.15),
    },
    site_names="Test",
)

result = atm.compute()
```

---

## SPARTA model parameters

The `SPARTA()` function can be called directly for single-call computation
without an atmosphere object.

```python
from spartasolar.modlib import SPARTA

out = SPARTA(
    cosz=0.7,
    pressure=1013.25,  # hPa
    pwater=2.0,        # cm
    ozone=0.3,         # atm-cm
    beta=0.1,
    alpha=1.3,
    ssa=0.92,
    asy=0.65,
    albedo=0.2,
    ecf=1.0,           # sun–earth distance correction factor
    csi_param="sparta",
    csi_hfov=2.5,      # pyrheliometer half-FOV, degrees
)

print(out.keys())  # ghi, dni, dif, dhi, csi, cosz
```

### Parameter ranges

| Parameter | Typical range | Units |
|---|---|---|
| `cosz` | 0 – 1 | – |
| `pressure` | 800 – 1100 | hPa |
| `pwater` | 0.1 – 7.0 | cm |
| `ozone` | 0.15 – 0.55 | atm-cm |
| `beta` | 0.01 – 0.5 | – |
| `alpha` | 0.5 – 2.0 | – |
| `ssa` | 0.80 – 0.99 | – |
| `albedo` | 0.0 – 1.0 | – |

---

## Configuration cheat-sheet

```python
from spartasolar import config

# View all settings
config.show_config()

# Read a value
val = config.get_option("merra2_daily.data_dir", default=None)

# Write a value
config.set_option("merra2_daily.data_dir", "/my/data")

# Reset to factory defaults
config.reset_config_file()
```

---

## Common xarray operations on outputs

```python
result = atm.compute()          # xarray.Dataset

# Select
result.ghi.sel(site="Málaga")
result.ghi.sel(time="2020-06-21 12:00")

# Aggregate
result.ghi.mean(dim="time")
result.ghi.resample(time="D").max()

# Include atmosphere variables in output
result_full = atm.compute(include_atmosphere=True)
print(result_full.data_vars)    # ghi, dni, dif, csi, pressure, pwater, …

# Export
result.to_netcdf("clear_sky.nc")
result.ghi.to_pandas()
```
