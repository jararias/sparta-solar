# pysparta

**Solar PArameterization of the Radiative Transfer of the Atmosphere**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-103%20passing-success)](tests/)

A high-performance Python library for computing clear-sky solar irradiance using the SPARTA radiative transfer model, with integrated access to multiple atmospheric databases.

---

## ✨ Features

- **🚀 High-Performance SPARTA Model**: State-of-the-art 2-band broadband clear-sky solar radiation model
- **🌍 Multiple Atmospheric Databases**: 
  - NASA MERRA-2 daily reanalysis (1980-present)
  - MERRA-2 long-term monthly averages (1999-2018)
  - Copernicus Radiation Service (CRS) via SODA API
  - Google Earth Engine MERRA-2 access
  - Custom user-defined atmospheres
- **📊 CF-Compliant Outputs**: xarray Datasets following Climate and Forecast metadata conventions
- **⚡ Vectorized Computations**: Efficiently handles both site-specific and gridded calculations
- **🔧 Flexible Configuration**: TOML-based configuration with sensible defaults
- **✅ Thoroughly Tested**: 103 unit tests with 100% pass rate
- **📚 Comprehensive Documentation**: NumPy-style docstrings with examples throughout

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Atmospheric Data Sources](#-atmospheric-data-sources)
- [Advanced Usage](#-advanced-usage)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [License](#-license)

---

## 🔧 Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/jararias/pysparta@main
```

### Development Installation

```bash
git clone https://github.com/jararias/pysparta.git
cd pysparta
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

### Basic Usage with MERRA-2 Daily Data

```python
import pandas as pd
from pysparta.atmoslib import MERRA2DailyAtmosphere

# Define time and location
times = pd.date_range("2020-06-15", periods=24, freq="h")
latitude = 36.72   # Málaga, Spain
longitude = -4.42

# Retrieve atmospheric data
atm = MERRA2DailyAtmosphere.at_sites(
    times=times,
    latitude=latitude,
    longitude=longitude,
    site_names="Málaga"
)

# Compute clear-sky solar radiation
result = atm.compute(model="SPARTA")

# Access results
print(result.ghi.values)  # Global Horizontal Irradiance
print(result.dni.values)  # Direct Normal Irradiance
print(result.dif.values)  # Diffuse Horizontal Irradiance
```

### Direct SPARTA Model Usage

```python
from pysparta.modlib import SPARTA

# Compute with explicit parameters
result = SPARTA(
    cosz=0.866,        # cos(30°) solar zenith angle
    pressure=1013.25,  # hPa
    pwater=2.0,        # cm
    ozone=0.3,         # atm-cm
    beta=0.1,          # Ångström turbidity
    alpha=1.3,         # Ångström exponent
    albedo=0.2         # surface albedo
)

print(f"DNI: {result['dni']:.1f} W/m²")
print(f"GHI: {result['ghi']:.1f} W/m²")
```

### Gridded Calculations

```python
import numpy as np

# Define spatial grid
lats = np.linspace(36.0, 41.0, 20)
lons = np.linspace(-5.0, -3.0, 20)
times = pd.date_range("2020-06-15 12:00", periods=1, freq="h")

# Get gridded atmospheric data
atm = MERRA2DailyAtmosphere.on_regular_grid(
    times=times,
    latitude=lats,
    longitude=lons
)

# Compute solar radiation
result = atm.compute(model="SPARTA")

# Visualize
result.ghi.isel(time=0).plot()
```

---

## 🧩 Core Concepts

### 1. Atmospheric Data Layer

pysparta separates atmospheric data retrieval from radiation calculations:

```python
# Step 1: Get atmospheric constituents (aerosols, water vapor, ozone, etc.)
from pysparta.atmoslib import MERRA2DailyAtmosphere

atm = MERRA2DailyAtmosphere.at_sites(
    times=times,
    latitude=36.72,
    longitude=-4.42
)

# Inspect available variables
print(atm.dataset.data_vars)
# Output: pressure, albedo, pwater, ozone, alpha, beta, ssa

# Step 2: Compute solar radiation
result = atm.compute(model="SPARTA")
```

### 2. Multiple Data Formats

**Site-based (time series at fixed locations):**
```python
atm = atmosphere.at_sites(
    times=times,
    latitude=[36.72, 40.42],      # Multiple sites
    longitude=[-4.42, -3.70],
    site_names=["Málaga", "Madrid"]
)
# Output dimensions: (time, site)
```

**Grid-based (spatial maps):**
```python
atm = atmosphere.on_regular_grid(
    times=times,
    latitude=np.arange(36, 41, 0.5),
    longitude=np.arange(-5, -3, 0.5)
)
# Output dimensions: (time, lat, lon)
```

### 3. CF-Compliant Output

All outputs follow Climate and Forecast metadata conventions:

```python
result = atm.compute(model="SPARTA")

# Rich metadata
print(result.ghi.attrs)
# {'standard_name': 'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky',
#  'long_name': 'clear-sky global horizontal irradiance',
#  'units': 'W m-2'}

# Easy export to NetCDF
result.to_netcdf("output.nc")
```

---

## 🌍 Atmospheric Data Sources

### MERRA2DailyAtmosphere (Recommended)

Daily NASA MERRA-2 reanalysis data (1980-present):

```python
from pysparta.atmoslib import MERRA2DailyAtmosphere

atm = MERRA2DailyAtmosphere.at_sites(
    times=pd.date_range("2020-01-01", "2020-12-31", freq="h"),
    latitude=36.72,
    longitude=-4.42
)
```

**Coverage**: Global | **Resolution**: 0.5° × 0.625° | **Temporal**: Hourly

### MERRA2LTAAtmosphere

Long-term monthly averages (1999-2018 climatology):

```python
from pysparta.atmoslib import MERRA2LTAAtmosphere

atm = MERRA2LTAAtmosphere.at_sites(
    times=pd.date_range("2023-01-01", periods=12, freq="MS"),
    latitude=36.72,
    longitude=-4.42
)
```

**Use case**: Typical meteorological year (TMY) generation

### CRSSODAAtmosphere

Copernicus Radiation Service via SODA API (requires registration):

```python
from pysparta import config
from pysparta.atmoslib import CRSSODAAtmosphere

# Configure user email (one-time setup)
config.set_option('crs_soda.user_email', 'your.email@example.com')

atm = CRSSODAAtmosphere.at_site(
    times=times,
    latitude=36.72,
    longitude=-4.42
)
```

**Coverage**: 2004-present | **Resolution**: 1-minute (resampled to hourly)

### MERRA2GEEAtmosphere

Access MERRA-2 via Google Earth Engine (requires GEE account):

```python
from pysparta import config
from pysparta.atmoslib import MERRA2GEEAtmosphere

# Configure GEE project
config.set_option('merra2_gee.project', 'your-gee-project-id')

atm = MERRA2GEEAtmosphere.at_site(
    times=times,
    latitude=36.72,
    longitude=-4.42
)
```

**Note**: Automatically corrects GEE's 0.25° latitude grid offset

### CustomAtmosphere

Use your own atmospheric measurements:

```python
import numpy as np
from pysparta.atmoslib import CustomAtmosphere

times = pd.date_range("2020-06-01", periods=24, freq="h")
constituents = {
    "pressure": np.full((24, 1), 101325.0),  # Pa
    "pwater": np.linspace(1.0, 3.0, 24).reshape(24, 1),  # cm
    "ozone": np.full((24, 1), 0.3),  # atm-cm
    "alpha": np.full((24, 1), 1.3),
    "beta": np.full((24, 1), 0.1),
    "albedo": np.full((24, 1), 0.2)
}

atm = CustomAtmosphere.at_sites(
    times=times,
    latitude=36.72,
    longitude=-4.42,
    constituents=constituents
)
```

---

## 🔬 Advanced Usage

### Model Comparison

```python
# Compare SPARTA with Bird clear-sky model
result_sparta = atm.compute(model="SPARTA")
result_bird = atm.compute(model="Bird")

# Calculate differences
diff = result_sparta.ghi - result_bird.ghi
```

### Circumsolar Irradiance

```python
from pysparta.modlib import SPARTA

# Enable circumsolar correction (default)
result = SPARTA(
    cosz=0.7,
    beta=0.3,  # High aerosol loading
    csi_param='sparta',  # Use SPARTA CSI parameterization
    csi_hfov=2.5  # Pyrheliometer half field-of-view (degrees)
)

csi_fraction = result['csi'] / result['dni']
print(f"CSI accounts for {csi_fraction:.2%} of DNI")
```

### Transmittance Schemes

```python
# Interdependent scheme (default, more accurate)
result_inter = SPARTA(
    cosz=0.8,
    transmittance_scheme='interdependent'
)

# Independent scheme (faster, less accurate)
result_indep = SPARTA(
    cosz=0.8,
    transmittance_scheme='independent'
)
```

### Custom Aerosol Properties

```python
result = SPARTA(
    cosz=0.866,
    beta=0.2,      # High turbidity
    alpha=0.8,     # Coarse particles (dust)
    ssa=0.88,      # Slightly absorbing
    asy=0.70       # Forward scattering
)
```

---

## ⚙️ Configuration

pysparta uses TOML-based configuration stored in `~/.config/pysparta/config.toml` (Linux) or equivalent on other platforms.

### View Configuration

```python
from pysparta import config

config.show_config()
```

### Set Options

```python
from pysparta import config

# Set custom data directory
config.set_option('merra2_daily.data_dir', '/data/merra2')

# Configure SODA user email
config.set_option('crs_soda.user_email', 'user@example.com')

# Set sunwhere algorithm
config.set_option('sunwhere.algorithm', 'nrel')
```

### Get Options

```python
algorithm = config.get_option('sunwhere.algorithm', default='psa')
data_dir = config.get_option('merra2_daily.data_dir')
```

### Reset to Defaults

```python
config.reset_config_file()
```

---

## ✅ Testing

pysparta includes a comprehensive test suite with 103 unit tests:

### Run All Tests

```bash
make tests
# or
pytest tests/
```

### Run Specific Test Modules

```bash
pytest tests/unit/test_sparta.py          # SPARTA model tests
pytest tests/unit/test_validation.py      # Validation framework tests
pytest tests/unit/test_config.py          # Configuration tests
pytest tests/unit/test_merra2_daily.py    # MERRA2 atmosphere tests
```

### Run with Coverage

```bash
pytest --cov=pysparta tests/
```

### Test Markers

```bash
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m slow          # Long-running tests
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

---

## 📚 Documentation

All modules include comprehensive NumPy-style docstrings with examples:

### Access Help

```python
from pysparta.modlib import SPARTA
help(SPARTA)

from pysparta.atmoslib import MERRA2DailyAtmosphere
help(MERRA2DailyAtmosphere.at_sites)
```

### Module Organization

```
pysparta/
├── modlib/              # Clear-sky radiation models
│   ├── sparta.py        # SPARTA model
│   └── bird.py          # Bird model
├── atmoslib/            # Atmospheric data sources
│   ├── _base.py         # Base classes and utilities
│   ├── merra2_daily.py  # MERRA-2 daily data
│   ├── merra2_lta.py    # MERRA-2 long-term averages
│   ├── merra2_geeapi.py # MERRA-2 via Google Earth Engine
│   ├── crs_sodaapi.py   # CRS SODA API access
│   └── custom.py        # Custom atmospheric data
├── validation.py        # Type validation framework
├── config.py            # Configuration management
└── logtools.py          # Logging utilities
```

---

## 📖 Citation

If you use pysparta in your research, please cite:

**Ruiz-Arias, J. A., & Arias, J. R.** (2025). Solar Parameterization of the Radiative Transfer of the Atmosphere (SPARTA): A two-band broadband clear-sky solar radiation model. *Solar Energy*, 280, 112836. https://doi.org/10.1016/j.solener.2024.112836

**BibTeX:**
```bibtex
@article{ruiz2025sparta,
  title={Solar Parameterization of the Radiative Transfer of the Atmosphere (SPARTA): A two-band broadband clear-sky solar radiation model},
  author={Ruiz-Arias, Jose A. and Arias, Javier R.},
  journal={Solar Energy},
  volume={280},
  pages={112836},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.solener.2024.112836}
}
```

---

## 📄 License

This project is licensed under the CC BY-NC-SA 4.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NASA GMAO for MERRA-2 reanalysis data
- Copernicus Atmosphere Monitoring Service (CAMS) for radiation service data
- Google Earth Engine for data infrastructure
- The xarray, pandas, numpy... and many more open source communities

---

## 🔗 Related Projects

- [sunwhere](https://github.com/jararias/sunwhere) - High-performance solar position calculations

---

## 📧 Contact

**Jose A. Ruiz-Arias**  
Universidad de Málaga, Spain  
Email: jararias@uma.es

---

**Made with ❤️ for the solar energy research community**
