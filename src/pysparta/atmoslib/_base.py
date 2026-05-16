r"""Base Abstract Classes for Atmospheric Data Handling.

This module defines the blueprint for all atmosphere database implementations.
It uses an abstract base class (ABC) to ensure a consistent interface for 
retrieving variables, coordinates, and elevation data across different 
atmospheric models.
"""

import abc
import inspect
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sunwhere
import xarray as xr
from loguru import logger

from .conversions import pwater_in_kg_m2_to_cm, ozone_in_kg_m2_to_cm
from .. import modlib, config
from ..validation import Latitude, Longitude, Model, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)

default_var_attrs = {
    "cosz": {
        "standard_name": "cosine_of_solar_zenith_angle",
        "long_name": "cosine of solar zenith angle",
        "units": "1"
    },
    "ghi": {
        "standard_name": "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",
        "long_name": "clear-sky global horizontal irradiance",
        "units": "W m-2"
    },
    "dni": {
        "standard_name": "surface_direct_downwelling_shortwave_flux_in_air_assuming_clear_sky",
        "long_name": "clear-sky direct normal irradiance",
        "units": "W m-2"
    },
    "dhi": {
        "long_name": "clear-sky direct horizontal irradiance",
        "units": "W m-2"
    },
    "dif": {
        "standard_name": "surface_diffuse_downwelling_shortwave_flux_in_air_assuming_clear_sky",
        "long_name": "clear-sky diffuse irradiance",
        "units": "W m-2"
    },
    "csi": {
        "long_name": "clear-sky circumsolar irradiance",
        "units": "W m-2"
    },
    "pressure": {
        "standard_name": "surface_air_pressure",
        "long_name": "surface pressure",
        "units": "Pa"
    },
    "albedo": {
        "standard_name": "surface_albedo",
        "long_name": "surface albedo",
        "units": "1"
    },
    "pwater": {
        "standard_name": "atmosphere_precipitable_water_content",
        "long_name": "precipitable water",
        "units": "kg m-2"
    },
    "ozone": {
        "standard_name": "atmosphere_total_column_ozone_content",
        "long_name": "total column ozone",
        "units": "kg m-2"
    },
    "beta": {
        "long_name": "Angstrom turbidity parameter",
        "units": "1"
    },
    "alpha": {
        "standard_name": "angstrom_exponent",
        "long_name": "Angstrom wavelength parameter",
        "units": "1"
    },
    "ssa": {
        "standard_name": "aerosol_single_scattering_albedo",
        "long_name": "aerosol single scattering albedo",
        "units": "1"
    },
    "altitude": {
        "standard_name": "surface_altitude",
        "long_name": "surface altitude",
        "units": "m"
    }
}

default_coord_attrs = {
    "lat": {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "axis": "Y"
    },
    "lon": {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "axis": "X"
    },
    "time": {
        "standard_name": "time",
        "long_name": "time",
        "axis": "T"
        # xarray gestiona las 'units' (e.g. 'seconds since...') automáticamente al guardar a NetCDF
    },
    "month": {
        "long_name": "month",
        "axis": "T"
    },
}

def get_global_attrs(
    title: str = "Clear-sky Atmospheric Dataset for SPARTA",
    institution: str = "Universidad de Málaga, Spain",
    source: str = "NASA/GMAO MERRA-2 reanalysis",
    references: str = "doi: 10.1016/j.rser.2023.113833",
    feature_type: str = "grid" # 'grid' si es una malla espacial, o 'timeSeries' si es una serie temporal de sitios puntuales
) -> dict:
    """Generate CF-compliant global attributes for atmospheric datasets.
    
    Creates a dictionary of global metadata attributes following CF conventions
    and ACDD (Attribute Convention for Data Discovery) recommendations.
    
    Parameters
    ----------
    title : str, default "Clear-sky Atmospheric Dataset for SPARTA"
        Concise description of the dataset
    institution : str, default "Universidad de Málaga, Spain"
        Institution responsible for producing the data
    source : str, default "NASA/GMAO MERRA-2 reanalysis"
        Method of production or data source
    references : str, default "doi: 10.1016/j.rser.2023.113833"
        Published or web-based references describing the data
    feature_type : str, default "grid"
        Type of data: 'grid' for gridded spatial data, 'timeSeries' for
        time series at fixed locations
        
    Returns
    -------
    dict
        Global attributes dictionary with CF/ACDD metadata
        
    Examples
    --------
    >>> attrs = get_global_attrs(
    ...     title="Custom Solar Dataset",
    ...     institution="My University",
    ...     feature_type="timeSeries"
    ... )
    >>> print(attrs["Conventions"])  # 'CF-1.11'
    
    See Also
    --------
    make_cf_compliant : Apply CF conventions to a dataset
    """
    from datetime import datetime, UTC
    return {
        "Conventions": "CF-1.11",
        "title": title,
        "history": f"Created on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "institution": institution,
        "source": source,
        "references": references,
        "featureType": feature_type,
        "creator_name": "Jose A Ruiz-Arias",
        "creator_email": "jararias at uma dot es",
    }


def make_cf_compliant(dataset: xr.Dataset, overwrite: bool = False) -> xr.Dataset:
    """Ensure xarray Dataset complies with CF conventions.

    This function validates and enriches dataset metadata to follow Climate and
    Forecast (CF) metadata conventions. It adds standard_name, long_name, and
    units attributes to coordinates and variables, and ensures proper global
    attributes are present.

    Parameters
    ----------
    dataset : xr.Dataset
        Input xarray Dataset to validate and modify
    overwrite : bool, default False
        If True, existing attributes will be replaced with CF-compliant values.
        If False, only missing attributes are added.
        
    Returns
    -------
    xr.Dataset
        CF-compliant dataset with enriched metadata
        
    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create a simple dataset
    >>> times = pd.date_range("2020-01-01", periods=10, freq="D")
    >>> ds = xr.Dataset(
    ...     {"ghi": (["time", "lat", "lon"], np.random.rand(10, 5, 5))},
    ...     coords={"time": times, "lat": np.arange(36, 41), "lon": np.arange(-5, 0)}
    ... )
    >>> ds_cf = make_cf_compliant(ds)
    >>> print(ds_cf.ghi.attrs["units"])  # 'W m-2'
    >>> print(ds_cf.lat.attrs["standard_name"])  # 'latitude'
    
    Notes
    -----
    CF conventions ensure datasets are self-describing and interoperable.
    See http://cfconventions.org/ for the full specification.
    
    See Also
    --------
    get_global_attrs : Generate CF-compliant global attributes
    """
    # Ensure coordinate variables have the correct attributes
    for coord in filter(lambda c: c in dataset.coords, ["time", "lat", "lon", "month"]):
        for attr_name, attr_value in default_coord_attrs[coord].items():
            if overwrite or attr_name not in dataset.coords[coord].attrs:
                dataset.coords[coord].attrs[attr_name] = attr_value
        if coord == "time":
            if "units" in dataset.coords[coord].encoding and "units" not in dataset.coords[coord].attrs:
                dataset.coords[coord].attrs["units"] = dataset.coords[coord].encoding["units"]
            if "calendar" in dataset.coords[coord].encoding and "calendar" not in dataset.coords[coord].attrs:
                dataset.coords[coord].attrs["calendar"] = dataset.coords[coord].encoding["calendar"]

    if "site_name" in dataset.coords:
        if "long_name" not in dataset.coords["site_name"].attrs:
            dataset.coords["site_name"].attrs["long_name"] = "name of the site"
        if "cf_role" not in dataset.coords["site_name"].attrs:
            dataset.coords["site_name"].attrs["cf_role"] = "timeseries_id"

    # Ensure data variables have the correct attributes
    for var in dataset.data_vars:
        if var in default_var_attrs:
            for attr_name, attr_value in default_var_attrs[var].items():
                if overwrite or attr_name not in dataset[var].attrs:
                    dataset[var].attrs[attr_name] = attr_value
        else:
            if "long_name" not in dataset[var].attrs:
                dataset[var].attrs["long_name"] = "unknown"
            if "units" not in dataset[var].attrs:
                dataset[var].attrs["units"] = "unknown"

    default_global_attrs = get_global_attrs(feature_type="timeSeries" if "site" in dataset.dims else "grid")
    for attr_name, attr_value in default_global_attrs.items():
        if overwrite or attr_name not in dataset.attrs:
            dataset.attrs[attr_name] = attr_value

    return dataset


def build_atmosphere_of_sites(
    times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
    latitude: Sequence[float] | float,
    longitude: Sequence[float] | float,
    constituents: dict[str, np.ndarray[tuple[int, int], float]],   # shape: (time, site)
    site_names: Sequence[str] | None = None,
    var_attrs: dict | None = None,
    global_attrs: dict | None = None,
) -> xr.DataArray:
    """Build a CF-compliant xarray Dataset for time series at specific sites.
    
    Constructs an xarray Dataset with dimensions (time, site) containing
    atmospheric constituent data at fixed locations.
    
    Parameters
    ----------
    times : np.ndarray or pd.DatetimeIndex
        Time stamps (length n_times)
    latitude : float or Sequence[float]
        Latitude(s) in degrees North (length n_sites)
    longitude : float or Sequence[float]
        Longitude(s) in degrees East (length n_sites)
    constituents : dict[str, np.ndarray]
        Atmospheric variables as 2D arrays (n_times, n_sites)
    site_names : Sequence[str], optional
        Names for each site
    var_attrs : dict, optional
        Additional variable attributes
    global_attrs : dict, optional
        Additional global attributes
        
    Returns
    -------
    xr.Dataset
        CF-compliant dataset with dimensions (time, site)
        
    Examples
    --------
    >>> times = pd.date_range("2020-01-01", periods=24, freq="h")
    >>> data = build_atmosphere_of_sites(
    ...     times=times,
    ...     latitude=[36.72, 40.42],
    ...     longitude=[-4.42, -3.70],
    ...     constituents={"pwater": np.random.rand(24, 2)},
    ...     site_names=["Málaga", "Madrid"]
    ... )
    """

    latitude = [latitude] if isinstance(latitude, (float, int)) else latitude
    latitude = np.asarray([validate_type(lat, Latitude) for lat in latitude], dtype=float).reshape(-1)

    longitude = [longitude] if isinstance(longitude, (float, int)) else longitude
    longitude = np.asarray([validate_type(lon, Longitude) for lon in longitude], dtype=float).reshape(-1)

    if len(latitude) != len(longitude):
        raise ValueError(f"incompatitble latitude (of length {len(latitude)}) and longitude (of "
                            f"length {len(longitude)}). They must have the same length")

    # parse datetimes, explicitly set them to UTC and remove timezone info (if any)
    times = pd.to_datetime(times, utc=True).tz_localize(tz=None)

    n_times = len(times)
    n_sites = len(latitude)

    dims = ("time", "site")
    coords = {"time": ("time", times), "lat": ("site", latitude), "lon": ("site", longitude)}

    if site_names is not None:
        coords.update(
            {"site_name": ("site", [site_names] if isinstance(site_names, str) else site_names)})

    data_vars = {}
    for constituent_name, constituent_values in constituents.items():
        values = np.asarray(constituent_values, dtype=float)
        if values.size != n_times * n_sites:
            logger.warning(f"`{constituent_name}` has size `{values.size}` which cannot be "
                           f"reshaped to ({n_times=}, {n_sites=}). Skipping")
            continue
        data_vars[constituent_name] = (dims, values.reshape((n_times, n_sites)))

    data = make_cf_compliant(xr.Dataset(data_vars, coords), overwrite=True)

    if var_attrs is not None:
        for var in filter(lambda var: var in data_vars, var_attrs):
            for attr_name, attr_value in var_attrs[var].items():
                data[var].attrs[attr_name] = attr_value

    if global_attrs is not None:
        for attr_name, attr_value in global_attrs.items():
            data.attrs[attr_name] = attr_value

    return data


def build_atmosphere_on_regular_grid(
    times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
    latitude: Sequence[float],
    longitude: Sequence[float],
    constituents: dict[str, np.ndarray[float]],
    var_attrs: dict | None = None,
    global_attrs: dict | None = None,
) -> xr.DataArray:
    """Build a CF-compliant xarray Dataset for data on a regular spatial grid.
    
    Constructs an xarray Dataset with dimensions (time, lat, lon) for gridded
    atmospheric constituent data.
    
    Parameters
    ----------
    times : np.ndarray or pd.DatetimeIndex
        Time stamps (length n_times)
    latitude : Sequence[float]
        Latitude coordinates in degrees North (length n_lats)
    longitude : Sequence[float]
        Longitude coordinates in degrees East (length n_lons)
    constituents : dict[str, np.ndarray]
        Atmospheric variables as 3D arrays (n_times, n_lats, n_lons)
    var_attrs : dict, optional
        Additional variable attributes
    global_attrs : dict, optional
        Additional global attributes
        
    Returns
    -------
    xr.Dataset
        CF-compliant dataset with dimensions (time, lat, lon)
        
    Examples
    --------
    >>> import numpy as np
    >>> times = pd.date_range("2020-01-01", periods=5, freq="h")
    >>> lats = np.arange(36, 41, 0.5)
    >>> lons = np.arange(-5, -3, 0.5)
    >>> data = build_atmosphere_on_regular_grid(
    ...     times=times,
    ...     latitude=lats,
    ...     longitude=lons,
    ...     constituents={"pwater": np.random.rand(5, len(lats), len(lons))}
    ... )
    """

    latitude = np.asarray([validate_type(lat, Latitude) for lat in latitude], dtype=float).reshape(-1)
    longitude = np.asarray([validate_type(lon, Longitude) for lon in longitude], dtype=float).reshape(-1)

    # parse datetimes, explicitly set them to UTC and remove timezone info (if any)
    times = pd.to_datetime(times, utc=True).tz_localize(tz=None)

    n_times = len(times)
    n_lats = len(latitude)
    n_lons = len(longitude)

    dims = ("time", "lat", "lon")
    coords = {"time": ("time", times), "lat": ("lat", latitude), "lon": ("lon", longitude)}

    data_vars = {}
    for constituent_name, constituent_values in constituents.items():
        values = np.asarray(constituent_values, dtype=float)
        if values.size != n_times * n_lats * n_lons:
            logger.warning(f"`{constituent_name}` has size `{values.size}` which cannot be "
                           f"reshaped to ({n_times=}, {n_lats=}, {n_lons=}). Skipping")
            continue
        data_vars[constituent_name] = (dims, values.reshape((n_times, n_lats, n_lons)))

    data = make_cf_compliant(xr.Dataset(data_vars, coords), overwrite=True)

    if var_attrs is not None:
        for var in filter(lambda var: var in data_vars, var_attrs):
            for attr_name, attr_value in var_attrs[var].items():
                data[var].attrs[attr_name] = attr_value

    if global_attrs is not None:
        for attr_name, attr_value in global_attrs.items():
            data.attrs[attr_name] = attr_value

    return data


class BaseAtmosphere(metaclass=abc.ABCMeta):
    """Abstract base class for atmospheric database interfaces.

    This class defines the common interface that all atmospheric data sources
    must implement. It provides methods for loading atmospheric constituent
    data and computing clear-sky solar radiation using various radiative
    transfer models.
    
    Subclasses must implement methods to retrieve atmospheric data either at
    specific sites (time series) or on regular grids. The class handles:
    - Loading and validation of atmospheric datasets
    - Integration with solar position calculations
    - Execution of clear-sky radiation models (SPARTA, Bird, etc.)
    - CF-compliant metadata management

    Attributes
    ----------
    database_path : Path or None
        Directory containing the database files (set by subclass)
    dataset : xr.Dataset
        The loaded atmospheric dataset
        
    Examples
    --------
    Subclasses should define `database_path` and implement data retrieval:
    
    >>> class MyAtmosphere(BaseAtmosphere, database_path="/path/to/data"):
    ...     @classmethod
    ...     def at_sites(cls, times, latitude, longitude, **kwargs):
    ...         # Implementation here
    ...         pass
    
    Notes
    -----
    This is an abstract base class and cannot be instantiated directly.
    Use concrete implementations like MERRA2DailyAtmosphere or CustomAtmosphere.
    
    See Also
    --------
    MERRA2DailyAtmosphere : Most commonly used implementation
    CustomAtmosphere : For user-provided data
    """

    def __init__(self):
        """Initialize the atmosphere instance.

        Validates that the database path exists (if specified) and initializes
        the internal atmosphere dataset to None.
        
        Raises
        ------
        AttributeError
            If database_path is specified but does not exist
        """
        if self.database_path is not None and not self.database_path.exists():
            raise AttributeError(f"missing path `{self.database_path}`")

        self._atmosphere: xr.DataArray = None

    def __init_subclass__(cls, database_path: str, **kwargs):
        """Automatically sets the database path for subclasses.

        Args:
            database_path: The directory path where the specific 
                atmosphere data is stored.
        """
        super().__init_subclass__(**kwargs)
        cls.database_path = None if database_path is None else Path(database_path)

    @property
    def dataset(self):
        return self._atmosphere

    def compute(
        self,
        model: Model = "SPARTA",
        include_atmosphere: bool = False,
        model_kwargs: dict | None = None,
    ) -> xr.Dataset:
        """Compute clear-sky solar radiation using a radiative transfer model.
        
        This method integrates solar position calculations with atmospheric
        constituent data to compute clear-sky irradiance components (GHI, DNI,
        DHI, etc.) using the specified radiative transfer model.
        
        Parameters
        ----------
        model : Model, default "SPARTA"
            Name of the clear-sky model to use. Options: "SPARTA", "Bird"
        include_atmosphere : bool, default False
            If True, include atmospheric constituents in the output dataset.
            If False, only radiation components are returned.
        model_kwargs : dict, optional
            Additional keyword arguments to pass to the model function
            
        Returns
        -------
        xr.Dataset
            CF-compliant dataset containing computed irradiance components:
            - ghi: Global Horizontal Irradiance (W/m²)
            - dni: Direct Normal Irradiance (W/m²)
            - dhi or dif: Diffuse Horizontal Irradiance (W/m²)
            - csi: Circumsolar Irradiance (W/m², SPARTA only)
            
        Examples
        --------
        >>> import pandas as pd
        >>> from pysparta.atmoslib import MERRA2DailyAtmosphere
        >>>
        >>> times = pd.date_range("2020-06-15", periods=24, freq="h")
        >>> atm = MERRA2DailyAtmosphere.at_sites(
        ...     times=times,
        ...     latitude=36.72,
        ...     longitude=-4.42
        ... )
        >>> result = atm.compute(model="SPARTA")
        >>> print(result.ghi.values)
        
        Use different model with custom parameters:
        
        >>> result = atm.compute(
        ...     model="Bird",
        ...     model_kwargs={"scheme": "transmittance_parameterization"}
        ... )
        
        Notes
        -----
        The method automatically:
        - Calculates solar position (zenith angle, Earth-Sun distance)
        - Converts atmospheric units to model requirements
        - Handles both gridded and site-based data structures
        
        See Also
        --------
        pysparta.modlib.sparta : SPARTA model implementation
        pysparta.modlib.bird : Bird clear-sky model
        """

        model = validate_type(model, Model)
        model_func = getattr(modlib, model)
        model_vars = inspect.getfullargspec(model_func).args

        is_regular_grid = "site" not in self.dataset.dims

        # compute solar geometry...
        sw_eval = sunwhere.regular_grid if is_regular_grid else sunwhere.sites
        solpos = sw_eval(
            self.dataset.time.values,
            latitude=self.dataset.lat.values,
            longitude=self.dataset.lon.values,
            algorithm=config.get_option("sunwhere.algorithm", default="psa"),
            refraction=config.get_option("sunwhere.refraction", default=True),
            engine=config.get_option("sunwhere.engine", default="numexpr")
        )

        new_coord_names = {"location": "site", "latitude": "lat", "longitude": "lon"}
        cosz = solpos.cosz.rename({old: new for old, new in new_coord_names.items() if old in solpos.cosz.dims})

        if is_regular_grid:
            n_lats = self.dataset.sizes["lat"]
            n_lons = self.dataset.sizes["lon"]
            ecf = solpos.ecf.expand_dims(dim={"lat": n_lats, "lon": n_lons}, axis=(1, 2))
        else:
            n_locs = self.dataset.sizes["site"]
            ecf = solpos.ecf.expand_dims(dim={"site": n_locs}, axis=1)

        kwargs = {"cosz": cosz, "ecf": ecf}

        # atmosphere...
        def get_with_proper_units(var):
            if var not in self.dataset.data_vars:
                raise ValueError(f"variable `{var}` required by the model `{model}` is not available in this atmosphere")
            if var == "pressure":
                return self.dataset[var] * 1e-2  # Pa to hPa
            if var == "pwater":
                return pwater_in_kg_m2_to_cm(self.dataset[var])
            if var == "ozone":
                return ozone_in_kg_m2_to_cm(self.dataset[var])
            return self.dataset[var]
        variables = set(model_vars).intersection(self.dataset.data_vars)  # variables required by this model
        kwargs = kwargs | {var: get_with_proper_units(var) for var in variables}

        # and call the clearsky model...
        result = model_func(**(kwargs | (model_kwargs or {})))

        # encapsulate the result in a CF-compliant xarray Dataset
        if is_regular_grid:
            return build_atmosphere_on_regular_grid(
                times=self.dataset.time.values,
                latitude=self.dataset.lat.values,
                longitude=self.dataset.lon.values,
                constituents=result,
                global_attrs=get_global_attrs(feature_type="grid")
            )
        else:
            return build_atmosphere_of_sites(
                times=self.dataset.time.values,
                latitude=self.dataset.lat.values,
                longitude=self.dataset.lon.values,
                constituents=result,
                site_names=self.dataset.coords.get("site_name", None).values if "site_name" in self.dataset.coords else None,
                global_attrs=get_global_attrs(feature_type="timeSeries")
            )
