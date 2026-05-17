"""Custom Atmospheric Data Interface.

This module allows users to provide their own atmospheric constituent data
for solar radiation modeling with SPARTA. This is useful when you have
atmospheric measurements or model outputs that are not part of the standard
databases provided by sparta-solar.

Examples
--------
Create custom atmospheric data for specific sites:

>>> import numpy as np
>>> import pandas as pd
>>> from spartasolar.atmoslib import CustomAtmosphere
>>>
>>> # Define time series and locations
>>> times = pd.date_range("2020-06-01", periods=24, freq="h")
>>> latitude = [36.72, 40.42]  # Málaga, Madrid
>>> longitude = [-4.42, -3.70]
>>>
>>> # Provide atmospheric constituents (shape: time x sites)
>>> constituents = {
...     "pressure": np.full((24, 2), 101325.0),  # Pa
...     "pwater": np.random.uniform(1.0, 3.0, (24, 2)),  # cm
...     "ozone": np.full((24, 2), 0.3),  # atm-cm
...     "alpha": np.full((24, 2), 1.3),  # Angstrom exponent
...     "beta": np.random.uniform(0.05, 0.15, (24, 2)),  # Angstrom turbidity
...     "ssa": np.full((24, 2), 0.92),  # single scattering albedo
...     "albedo": np.full((24, 2), 0.2)  # surface albedo
... }
>>>
>>> atm = CustomAtmosphere.at_sites(
...     times=times,
...     latitude=latitude,
...     longitude=longitude,
...     constituents=constituents,
...     site_names=["Málaga", "Madrid"]
... )
>>>
>>> # Compute solar radiation
>>> result = atm.compute(model="SPARTA")
>>> print(result.ghi.values)

Create custom atmospheric data on a regular grid:

>>> # Define spatial grid
>>> lats = np.linspace(36.0, 41.0, 10)
>>> lons = np.linspace(-5.0, -3.0, 10)
>>> times = pd.date_range("2020-06-15 12:00", periods=1, freq="h")
>>>
>>> # Constituents shape: (time, lat, lon)
>>> constituents = {
...     "pressure": np.full((1, 10, 10), 101325.0),
...     "pwater": np.random.uniform(1.5, 2.5, (1, 10, 10)),
...     "ozone": np.full((1, 10, 10), 0.3),
...     "alpha": np.full((1, 10, 10), 1.3),
...     "beta": np.random.uniform(0.08, 0.12, (1, 10, 10)),
... }
>>>
>>> atm = CustomAtmosphere.on_regular_grid(
...     times=times,
...     latitude=lats,
...     longitude=lons,
...     constituents=constituents
... )

See Also
--------
MERRA2DailyAtmosphere : Use NASA MERRA-2 reanalysis data
BaseAtmosphere : Base class for all atmospheric databases
"""

from typing import Sequence, Self

import numpy as np
import pandas as pd
from loguru import logger

from ._base import BaseAtmosphere, build_atmosphere_of_sites, build_atmosphere_on_regular_grid

logger.disable(__name__)
logger = logger.opt(colors=True)


class CustomAtmosphere(
    BaseAtmosphere,
    database_path=None
):
    """Custom atmospheric data provider.
    
    This class allows users to supply their own atmospheric constituent data
    from measurements, numerical weather prediction models, or other sources.
    Data can be provided for specific sites (time series) or regular grids.
    
    See module documentation for examples.
    """

    @classmethod
    def at_sites(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float] | float,
        longitude: Sequence[float] | float,
        constituents: dict[str, np.ndarray[tuple[int, int], float]],  # shape: (time, site)
        site_names: Sequence[str] | None = None,
        var_attrs: dict | None = None,
        global_attrs: dict | None = None,
    ) -> Self:
        """Create custom atmospheric data for specific sites.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps for the data (length n_times)
        latitude : float or Sequence[float]
            Latitude(s) in degrees North [-90, 90] (length n_sites)
        longitude : float or Sequence[float]
            Longitude(s) in degrees East [-180, 180] (length n_sites)
        constituents : dict[str, np.ndarray]
            Atmospheric variables as 2D arrays with shape (n_times, n_sites).
            Standard variable names: 'pressure' (Pa), 'pwater' (cm), 'ozone' (atm-cm),
            'alpha', 'beta', 'ssa', 'albedo'
        site_names : Sequence[str], optional
            Names for each site
        var_attrs : dict, optional
            Custom attributes for variables (CF conventions)
        global_attrs : dict, optional
            Custom global attributes for the dataset
            
        Returns
        -------
        CustomAtmosphere
            Instance with atmospheric data loaded
            
        Examples
        --------
        >>> times = pd.date_range("2020-01-01", periods=24, freq="h")
        >>> atm = CustomAtmosphere.at_sites(
        ...     times=times,
        ...     latitude=36.72,
        ...     longitude=-4.42,
        ...     constituents={
        ...         "pressure": np.full((24, 1), 101325.0),
        ...         "pwater": np.linspace(1.0, 2.0, 24).reshape(24, 1),
        ...         "ozone": np.full((24, 1), 0.3)
        ...     }
        ... )
        """

        obj = cls()
        obj._atmosphere = build_atmosphere_of_sites(
            times=times,
            latitude=latitude,
            longitude=longitude,
            constituents=constituents,
            site_names=site_names,
            var_attrs=var_attrs,
            global_attrs=global_attrs)
        return obj

    @classmethod
    def on_regular_grid(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
        constituents: dict[str, np.ndarray[tuple[int, int, int], float]],  # shape: (time, lat, lon)
        var_attrs: dict | None = None,
        global_attrs: dict | None = None,
    ) -> Self:
        """Create custom atmospheric data on a regular spatial grid.
        
        Parameters
        ----------
        times : np.ndarray or pd.DatetimeIndex
            Time stamps for the data (length n_times)
        latitude : Sequence[float]
            Latitude coordinates in degrees North (length n_lats)
        longitude : Sequence[float]
            Longitude coordinates in degrees East (length n_lons)
        constituents : dict[str, np.ndarray]
            Atmospheric variables as 3D arrays with shape (n_times, n_lats, n_lons).
            Standard names: 'pressure', 'pwater', 'ozone', 'alpha', 'beta', 'ssa', 'albedo'
        var_attrs : dict, optional
            Custom variable attributes
        global_attrs : dict, optional
            Custom global dataset attributes
            
        Returns
        -------
        CustomAtmosphere
            Instance with gridded atmospheric data
            
        Examples
        --------
        >>> lats = np.linspace(36.0, 41.0, 20)
        >>> lons = np.linspace(-5.0, -3.0, 20)
        >>> times = pd.date_range("2020-06-15", periods=5, freq="h")
        >>> atm = CustomAtmosphere.on_regular_grid(
        ...     times=times,
        ...     latitude=lats,
        ...     longitude=lons,
        ...     constituents={
        ...         "pressure": np.full((5, 20, 20), 101325.0),
        ...         "pwater": np.random.uniform(1.0, 3.0, (5, 20, 20)),
        ...     }
        ... )
        """
    
        obj = cls()
        obj._atmosphere = build_atmosphere_on_regular_grid(
            times=times,
            latitude=latitude,
            longitude=longitude,
            constituents=constituents,
            var_attrs=var_attrs,
            global_attrs=global_attrs)
        return obj

