# Quickstart 🚀

Pysparta is designed to handle the complexity of solar radiation modeling with minimal effort. This guide will show you how to go from a geographic coordinate to a complete solar resource evaluation.

## 1. Setup your Credentials
Pysparta can automatically retrieve high-quality atmospheric data. To enable this, you need to provide your SoDA email once:

```python
import pysparta
pysparta.set_option("soda_user_email", "your.email@example.com")
```

## 2. Basic Point Simulation
The `sites()` function is the primary entry point. It orchestrates solar geometry, atmospheric data retrieval, and the radiative transfer model.

```python
import pysparta
import pandas as pd
import matplotlib.pyplot as plt

# Define time range (UTC) and location
times = pd.date_range("2023-06-21 05:00", "2023-06-21 21:00", freq="15min", tz="UTC")
lat, lon = 40.41, -3.70

# Run simulation using MERRA-2 climatology and SPARTA model
ds = pysparta.sites(
    times_utc=times,
    latitude=lat,
    longitude=lon,
    atmos="merra2_lta",
    model="SPARTA"
)

# Plot the results directly from the xarray Dataset
ds[["dni", "dif", "ghi"]].to_array().plot(hue="variable")
plt.title(f"Clear-Sky Irradiance at {lat}N, {lon}E")
plt.ylabel("Irradiance [W/m²]")
plt.show()
```

## 3. Spatial Grid Simulation
To evaluate an entire region, use `regular_grid()`. It generates a 3D/4D dataset (time, latitude, longitude).

```python
# Define a 1-degree resolution grid for Southern Spain
lats = [36.0, 37.0, 38.0]
lons = [-6.0, -5.0, -4.0]

grid_ds = pysparta.regular_grid(
    times_utc="2023-06-21 12:00",
    latitude=lats,
    longitude=lons,
    atmos="merra2_lta"
)

grid_ds["ghi"].plot()
```
