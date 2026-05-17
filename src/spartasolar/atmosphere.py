"""Convenience module to import all atmosphere classes from spartasolar.atmoslib.

It provides access to various atmospheric databases that supply input
parameters for clear-sky solar radiation modeling. Each database provides
atmospheric constituents (aerosol optical depth, water vapor, ozone, etc.)
from different sources:

- **MERRA2DailyAtmosphere** (merra2_daily): NASA MERRA-2 daily reanalysis (1980-present)
- **MERRA2LTAAtmosphere** (merra2_lta): MERRA-2 long-term monthly averages (1999-2018)
- **MERRA2CDAAtmosphere** (merra2_cda): MERRA-2 climate data archive
- **MERRA2GEEAtmosphere** (merra2_gee): MERRA-2 via Google Earth Engine API
- **CRSSODAAtmosphere** (crs_soda): CRS SODA API historical data
- **CustomAtmosphere** (custom): User-defined atmospheric data

Examples
--------

>>> from spartasolar.atmosphere import merra2_daily
>>> import pandas as pd
>>>
>>> # Use MERRA-2 daily reanalysis
>>> times = pd.date_range("2020-01-01", "2020-01-31", freq="h")
>>> latitude = [36.72, 40.42]  # Málaga, Madrid
>>> longitude = [-4.42, -3.70]
>>> data = merra2_daily.at_sites(
...     times=times,
...     latitude=latitude,
...     longitude=longitude,
...     site_names=["Málaga", "Madrid"]
... )
>>>
>>> # Access atmospheric variables
>>> print(data.dataset.data_vars)
>>> # Compute solar radiation
>>> result = data.compute(model="SPARTA")
>>> print(result)
"""

from .atmoslib.crs_sodaapi import CRSSODAAtmosphere
from .atmoslib.custom import CustomAtmosphere
from .atmoslib.merra2_geeapi import MERRA2GEEAtmosphere
from .atmoslib.merra2_cda import MERRA2CDAAtmosphere
from .atmoslib.merra2_lta import MERRA2LTAAtmosphere
from .atmoslib.merra2_daily import MERRA2DailyAtmosphere

custom = CustomAtmosphere()
crs_soda = CRSSODAAtmosphere()
merra2_gee = MERRA2GEEAtmosphere()
merra2_cda = MERRA2CDAAtmosphere()
merra2_lta = MERRA2LTAAtmosphere()
merra2_daily = MERRA2DailyAtmosphere()
