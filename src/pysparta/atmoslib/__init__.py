"""Atmospheric Data Sources for SPARTA Modeling.

This module provides access to various atmospheric databases that supply input
parameters for clear-sky solar radiation modeling. Each database provides
atmospheric constituents (aerosol optical depth, water vapor, ozone, etc.)
from different sources:

- **MERRA2DailyAtmosphere**: NASA MERRA-2 daily reanalysis (1980-present)
- **MERRA2LTAAtmosphere**: MERRA-2 long-term monthly averages (1999-2018)
- **MERRA2CDAAtmosphere**: MERRA-2 climate data archive
- **MERRA2GEEAtmosphere**: MERRA-2 via Google Earth Engine API
- **CRSSODAAtmosphere**: CRS SODA API historical data
- **CustomAtmosphere**: User-defined atmospheric data

Examples
--------
Access atmospheric databases through the Atmosphere container:

>>> from pysparta.atmoslib import Atmosphere
>>> import pandas as pd
>>>
>>> # Use MERRA-2 daily reanalysis
>>> atm = Atmosphere()
>>> times = pd.date_range("2020-01-01", "2020-01-31", freq="h")
>>> latitude = [36.72, 40.42]  # Málaga, Madrid
>>> longitude = [-4.42, -3.70]
>>> data = atm.merra2_daily.at_sites(
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

Or use individual atmosphere classes directly:

>>> from pysparta.atmoslib import MERRA2DailyAtmosphere
>>> atm = MERRA2DailyAtmosphere.at_sites(
...     times=times,
...     latitude=36.72,
...     longitude=-4.42
... )

See Also
--------
MERRA2DailyAtmosphere : Most commonly used, daily temporal resolution
CustomAtmosphere : For using custom atmospheric data
"""

import dataclasses

from .crs_sodaapi import CRSSODAAtmosphere
from .custom import CustomAtmosphere
from .merra2_geeapi import MERRA2GEEAtmosphere
from .merra2_cda import MERRA2CDAAtmosphere
from .merra2_lta import MERRA2LTAAtmosphere
from .merra2_daily import MERRA2DailyAtmosphere

custom_atmosphere = CustomAtmosphere()

@dataclasses.dataclass
class Atmosphere:
    """Container for all available atmospheric data sources.
    
    Attributes
    ----------
    crs_soda : CRSSODAAtmosphere
        CRS SODA API atmospheric database
    merra2_gee : MERRA2GEEAtmosphere
        MERRA-2 via Google Earth Engine
    merra2_cda : MERRA2CDAAtmosphere
        MERRA-2 Climate Data Archive
    merra2_lta : MERRA2LTAAtmosphere
        MERRA-2 long-term monthly averages
    merra2_daily : MERRA2DailyAtmosphere
        MERRA-2 daily reanalysis (recommended)
    """
    crs_soda = CRSSODAAtmosphere()
    merra2_gee = MERRA2GEEAtmosphere()
    merra2_cda = MERRA2CDAAtmosphere()
    merra2_lta = MERRA2LTAAtmosphere()
    merra2_daily = MERRA2DailyAtmosphere()
