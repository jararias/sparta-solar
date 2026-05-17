"""MERRA-2 Long-Term Monthly Average Atmospheric Data.

This module provides access to NASA MERRA-2 long-term monthly average (LTA)
atmospheric constituent data computed over the period 1999-2018. This dataset
is useful for typical meteorological year (TMY) generation or when multi-year
averages are needed.

The LTA data includes monthly climatological values of:
- Aerosol optical properties (alpha, beta, ssa)
- Water vapor (pwater)
- Ozone (ozone)
- Surface pressure
- Surface albedo

Spatial resolution: 0.5° × 0.625° (latitude × longitude)
Temporal resolution: Monthly climatology (12 months)

Examples
--------
Retrieve long-term monthly averages for specific sites:

>>> import pandas as pd
>>> from pysparta.atmoslib import MERRA2LTAAtmosphere
>>>
>>> # Generate times covering multiple years
>>> times = pd.date_range("2020-01-15", "2021-12-15", freq="MS") + pd.Timedelta(14.5, "d")
>>> atm = MERRA2LTAAtmosphere.at_sites(
...     times=times,
...     latitude=[36.72, 40.42],
...     longitude=[-4.42, -3.70],
...     site_names=["Málaga", "Madrid"]
... )
>>> print(atm.dataset)

Retrieve data on a regular grid:

>>> import numpy as np
>>> lats = np.arange(36.0, 41.0, 0.5)
>>> lons = np.arange(-5.0, -3.0, 0.5)
>>> times = pd.date_range("2023-01-15", periods=12, freq="MS") + pd.Timedelta(14.5, "d")
>>> atm = MERRA2LTAAtmosphere.on_regular_grid(
...     times=times,
...     latitude=lats,
...     longitude=lons
... )

Notes
-----
The monthly climatology is repeated for each year in the requested time range.
Temporal interpolation uses quadratic splines for smooth transitions between months.

See Also
--------
MERRA2DailyAtmosphere : Daily temporal resolution with full historical coverage

References
----------
.. [1] Gelaro et al. (2017), The Modern-Era Retrospective Analysis for Research
       and Applications, Version 2 (MERRA-2). J. Climate, 30, 5419-5454.
"""

import importlib
from pathlib import Path
from typing import Self, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from loguru import logger

from ..validation import Latitude, Longitude, validate_type
from ._base import BaseAtmosphere, build_atmosphere_of_sites, build_atmosphere_on_regular_grid

logger.disable(__name__)
logger = logger.opt(colors=True)


def get_database_path():
    """Get the path to the bundled MERRA-2 LTA database.
    
    Returns
    -------
    Path
        Directory containing merra2_lta_data files
    """
    return Path(importlib.resources.files("pysparta.atmoslib")) / "merra2_lta_data"


class MERRA2LTAAtmosphere(
    BaseAtmosphere,
    database_path=get_database_path()
):
    """MERRA-2 long-term monthly average atmospheric database.
    
    Provides climatological monthly averages (1999-2018) of atmospheric
    constituents from NASA MERRA-2 reanalysis. Data is interpolated
    spatially and temporally to match requested coordinates and times.
    
    See module documentation for examples.
    """

    @classmethod
    def at_sites(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float] | float,
        longitude: Sequence[float] | float,
        site_names: Sequence[str] | None = None,
    ) -> Self:
        """Retrieve monthly climatology at specific sites.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps for climatology retrieval. Monthly climatology is
            repeated for each year and interpolated to exact times.
        latitude : Sequence[float]
            Latitude(s) in degrees North [-90, 90]
        longitude : Sequence[float]
            Longitude(s) in degrees East [-180, 180]
        site_names : Sequence[str], optional
            Names for each site
            
        Returns
        -------
        MERRA2LTAAtmosphere
            Instance with interpolated climatological data
            
        Examples
        --------
        >>> times = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        >>> atm = MERRA2LTAAtmosphere.at_sites(
        ...     times=times,
        ...     latitude=36.72,
        ...     longitude=-4.42,
        ...     site_names="Málaga"
        ... )
        """

        latitude = np.asarray(latitude, dtype=float).reshape(-1)
        latitude = [validate_type(lat, Latitude) for lat in latitude]
        longitude = np.asarray(longitude, dtype=float).reshape(-1)
        longitude = [validate_type(lon, Longitude) for lon in longitude]

        if len(latitude) != len(longitude):
            raise ValueError('latitude and longitude must have the same length')

        # load the dataset. Check for local availability. If not available, download.
        dataset = cls._load_dataset(times)

        # lat-lon interpolation
        output_lat = xr.DataArray(latitude, dims="site", name="lat")
        output_lon = xr.DataArray(longitude, dims="site", name="lon")
        output_dataset = dataset.interp(lat=output_lat, lon=output_lon, method='linear')

        # time interpolation
        if "time" in output_dataset.coords:
            output_dataset = output_dataset.interp(time=times, method='quadratic')

        global_attrs = {
            "title": "Long Term Average Clear-sky Atmospheric Dataset for SPARTA",
            "references": "doi:10.5067/KLICLTZ8EM9D, doi:10.5067/Q9QMY5PBNV1T, doi:10.5067/VJAFPLI1CSIV",
        }

        obj = cls()
        obj._atmosphere = build_atmosphere_of_sites(
            times=times,
            latitude=latitude,
            longitude=longitude,
            constituents=output_dataset.data_vars,
            site_names=site_names,
            global_attrs=global_attrs)
        return obj

    @classmethod
    def on_regular_grid(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
    ) -> Self:
        """Retrieve monthly climatology on a regular spatial grid.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps for climatology retrieval
        latitude : Sequence[float]
            Latitude grid coordinates in degrees North
        longitude : Sequence[float]
            Longitude grid coordinates in degrees East
            
        Returns
        -------
        MERRA2LTAAtmosphere
            Instance with gridded climatological data
            
        Examples
        --------
        >>> import numpy as np
        >>> lats = np.linspace(36.0, 41.0, 20)
        >>> lons = np.linspace(-5.0, -3.0, 20)
        >>> times = pd.date_range("2023-01-15", periods=12, freq="MS") + pd.Timedelta(14.5, "d")
        >>> atm = MERRA2LTAAtmosphere.on_regular_grid(
        ...     times=times,
        ...     latitude=lats,
        ...     longitude=lons
        ... )
        """

        latitude = np.asarray(latitude, dtype=float).reshape(-1)
        latitude = [validate_type(lat, Latitude) for lat in latitude]
        longitude = np.asarray(longitude, dtype=float).reshape(-1)
        longitude = [validate_type(lon, Longitude) for lon in longitude]

        # load the dataset. Check for local availability. If not available, download.
        dataset = cls._load_dataset(times)

        # lat-lon interpolation
        output_dataset = dataset.interp(lat=latitude, lon=longitude, method='linear')

        # time interpolation
        if "time" in output_dataset.coords:
            output_dataset = output_dataset.interp(time=times, method='quadratic')

        global_attrs = {
            "title": "Long Term Average Clear-sky Atmospheric Dataset for SPARTA",
            "references": "doi:10.5067/KLICLTZ8EM9D, doi:10.5067/Q9QMY5PBNV1T, doi:10.5067/VJAFPLI1CSIV",
        }

        obj = cls()
        obj._atmosphere = build_atmosphere_on_regular_grid(
            times=times,
            latitude=latitude,
            longitude=longitude,
            constituents=output_dataset.data_vars,
            global_attrs=global_attrs)
        return obj

    @staticmethod
    def _infer_years_from_times(times: npt.NDArray[np.datetime64] | pd.DatetimeIndex) -> list[int]:
        """Infer which years of climatology are needed based on requested times.
        
        Pads with extra years if times are near year boundaries to ensure
        smooth temporal interpolation.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps to analyze
            
        Returns
        -------
        list[int]
            Sorted list of years needed (may include padding years)
            
        Examples
        --------
        >>> times = pd.date_range("2020-06-01", "2020-08-31", freq="D")
        >>> years = MERRA2LTAAtmosphere._infer_years_from_times(times)
        >>> print(years)  # [2020]
        
        >>> # Near year boundary - includes padding year
        >>> times = pd.date_range("2020-01-01", "2020-01-05", freq="D")
        >>> years = MERRA2LTAAtmosphere._infer_years_from_times(times)
        >>> print(years)  # [2019, 2020]
        """
        the_times = times if isinstance(times, pd.DatetimeIndex) else pd.to_datetime(times)
        years = set(the_times.year)
        if (the_times[0] - pd.to_datetime(f"{min(years)}-01-15 12")) < pd.Timedelta(3, "D"):
            years.add(min(years)-1)
        if (pd.to_datetime(f"{max(years)}-12-15 12") - the_times[-1]) < pd.Timedelta(3, "D"):
            years.add(max(years)+1)
        return sorted(years)

    @classmethod
    def _load_dataset(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
    ) -> xr.Dataset:
        """Load and prepare MERRA-2 LTA dataset for requested time range.
        
        Loads the monthly climatology and replicates it for each year needed.
        Each monthly value is assigned to the 15th of the month at noon.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps to cover
            
        Returns
        -------
        xr.Dataset
            Dataset with monthly climatology expanded to cover all needed years
            
        Notes
        -----
        The dataset is chunked monthly for efficient I/O operations.
        """

        def assign_year(ds, year):
            times_month_start = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
            return ds.assign_coords(month=times_month_start + pd.Timedelta(14.5, "d")).rename({"month": "time"})

        years = cls._infer_years_from_times(times)
        dataset_lta = xr.open_dataset(cls.database_path / "merra2_lta_data_1999-2018.nc", chunks={})
        dataset = xr.concat([assign_year(dataset_lta, year) for year in years], dim="time", data_vars="minimal")

        if "time" in dataset.coords:
            return dataset.chunk(time=12)  # monthly chunks
        return dataset
