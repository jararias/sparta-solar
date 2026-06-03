"""MERRA-2 Climate Data Archive (CDA) Atmospheric Data.

This module provides a simplified interface to MERRA-2 long-term average data
with conservative default values for water vapor and aerosol optical depth.
It's designed for quick climate assessments where detailed temporal variability
is not critical.

The CDA atmosphere uses MERRA-2 LTA as base and applies:
- Water vapor: Fixed at 0.1 cm (very dry atmosphere)
- Aerosol turbidity (beta): Fixed at 0.01 (very clean atmosphere)
- Other variables: From MERRA-2 LTA climatology

This is useful for:
- Conservative solar energy estimates (low aerosol/water vapor)
- Climatological studies with simplified assumptions
- Quick assessments without detailed atmospheric data

Examples
--------
Use conservative atmospheric values for solar assessment:

>>> import pandas as pd
>>> from spartasolar.atmoslib import MERRA2CDAAtmosphere
>>>
>>> times = pd.date_range("2023-01-01", "2023-12-31", freq="D")
>>> atm = MERRA2CDAAtmosphere.at_sites(
...     times=times,
...     latitude=[36.72, 40.42],
...     longitude=[-4.42, -3.70],
...     site_names=["Málaga", "Madrid"]
... )
>>> print(atm.dataset.pwater.values)  # All 0.1 cm
>>> print(atm.dataset.beta.values)    # All 0.01

Notes
-----
This is a conservative approach that may overestimate clear-sky irradiance
due to the very low water vapor and aerosol loading.

See Also
--------
MERRA2LTAAtmosphere : Full LTA climatology without fixed values
MERRA2DailyAtmosphere : Daily temporal resolution with realistic variability
"""

from typing import Self, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from .merra2_lta import MERRA2LTAAtmosphere, get_database_path as lta_database_path

logger.disable(__name__)


def get_database_path():
    """Get the path to MERRA-2 LTA database (shared with CDA).
    
    Returns
    -------
    Path
        Directory containing MERRA-2 LTA data files
    """
    return lta_database_path()


class MERRA2CDAAtmosphere(
    MERRA2LTAAtmosphere,
    database_path=get_database_path()):
    """MERRA-2 Climate Data Archive with conservative default values.
    
    Extends MERRA2LTAAtmosphere with fixed conservative values for
    water vapor (0.1 cm) and aerosol turbidity (beta=0.01).
    
    See module documentation for examples and use cases.
    """

    @classmethod
    def at_sites(
        cls,
        times: np.ndarray[tuple[int], np.dtype[np.datetime64]] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
        site_names: Sequence[str] | None = None,
    ) -> Self:
        """Retrieve CDA atmospheric data at specific sites.
        
        Gets MERRA-2 LTA climatology and overrides water vapor and aerosol
        with conservative fixed values.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps for data retrieval
        latitude : Sequence[float]
            Latitude(s) in degrees North [-90, 90]
        longitude : Sequence[float]
            Longitude(s) in degrees East [-180, 180]
        site_names : Sequence[str], optional
            Names for each site
            
        Returns
        -------
        MERRA2CDAAtmosphere
            Instance with conservative atmospheric values
            
        Examples
        --------
        >>> times = pd.date_range("2023-06-01", periods=30, freq="D")
        >>> atm = MERRA2CDAAtmosphere.at_sites(
        ...     times=times,
        ...     latitude=36.72,
        ...     longitude=-4.42
        ... )
        """

        lta_atmos = super().at_sites(
            times=times,
            latitude=latitude,
            longitude=longitude,
            site_names=site_names)

        if "pwater" in lta_atmos.dataset:
            lta_atmos.dataset["pwater"] = 0.1

        if "beta" in lta_atmos.dataset:
            lta_atmos.dataset["beta"] = 0.01

        lta_atmos.dataset.attrs["title"] = "Clean and Dry, LTA Atmospheric Dataset for SPARTA"
        return lta_atmos

    @classmethod
    def on_regular_grid(
        cls,
        times: np.ndarray[tuple[int], np.dtype[np.datetime64]] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
    ) -> Self:
        """Retrieve CDA atmospheric data on a regular spatial grid.
        
        Gets MERRA-2 LTA climatology and overrides water vapor and aerosol
        with conservative fixed values.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps for data retrieval
        latitude : Sequence[float]
            Latitude grid coordinates in degrees North
        longitude : Sequence[float]
            Longitude grid coordinates in degrees East
            
        Returns
        -------
        MERRA2CDAAtmosphere
            Instance with gridded conservative atmospheric data
            
        Examples
        --------
        >>> lats = np.arange(36.0, 41.0, 0.5)
        >>> lons = np.arange(-5.0, -3.0, 0.5)
        >>> times = pd.date_range("2023-01-01", periods=12, freq="MS")
        >>> atm = MERRA2CDAAtmosphere.on_regular_grid(
        ...     times=times,
        ...     latitude=lats,
        ...     longitude=lons
        ... )
        """

        lta_atmos = super().on_regular_grid(
            times=times,
            latitude=latitude,
            longitude=longitude)

        if "pwater" in lta_atmos.dataset:
            lta_atmos.dataset["pwater"] = 0.1

        if "beta" in lta_atmos.dataset:
            lta_atmos.dataset["beta"] = 0.01

        lta_atmos.dataset.attrs["title"] = "Clean and Dry, LTA Atmospheric Dataset for SPARTA"
        return lta_atmos
