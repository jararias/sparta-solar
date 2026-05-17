"""MERRA-2 Atmospheric Data via Google Earth Engine API.

This module provides access to NASA MERRA-2 reanalysis atmospheric data
through Google Earth Engine (GEE). It retrieves hourly atmospheric constituents
including aerosol properties, water vapor, ozone, and surface parameters.

Data is accessed from three GEE collections:
- NASA/GSFC/MERRA/aer/2: Aerosol optical properties (AOD, Ångström parameters, SSA)
- NASA/GSFC/MERRA/slv/2: Single-level diagnostics (pressure, ozone, water vapor)
- NASA/GSFC/MERRA/rad/2: Radiation parameters (surface albedo)

Data availability: 1980-01-01 to near real-time (updated with ~2-3 week lag)
Temporal resolution: Hourly (time-averaged)
Spatial resolution: 0.5° × 0.625° (latitude × longitude)

Requirements
------------
- Google Earth Engine account (https://earthengine.google.com/)
- GEE Python API authentication
- Active GEE project

Examples
--------
Retrieve MERRA-2 data via GEE for a specific site:

>>> import pandas as pd
>>> from pysparta.atmoslib import MERRA2GEEAtmosphere
>>> from pysparta import config
>>>
>>> # Configure GEE project
>>> config.set_option('merra2_gee.project', 'your-gee-project-id')
>>>
>>> times = pd.date_range("2020-06-15", periods=24, freq="h")
>>> atm = MERRA2GEEAtmosphere.at_site(
...     times=times,
...     latitude=36.72,
...     longitude=-4.42,
...     site_name="Málaga"
... )
>>> print(atm.dataset)

Notes
-----
Important spatial correction:
GEE's MERRA-2 ingestion shifted the latitude grid by 0.25° northward.
This module automatically corrects for this offset by subtracting 0.25°
from query latitudes to retrieve the correct NASA grid cell.

The time stamps in GEE are adjusted to hour start (00:00), while NASA
uses hour center (00:30). This module corrects to NASA convention.

See Also
--------
MERRA2DailyAtmosphere : Direct access to MERRA-2 via local Zarr files
CRSSODAAtmosphere : Alternative using SODA API

References
----------
.. [1] Gelaro et al. (2017), The Modern-Era Retrospective Analysis for
       Research and Applications, Version 2 (MERRA-2). J. Climate, 30, 5419-5454.
.. [2] Gorelick et al. (2017), Google Earth Engine: Planetary-scale geospatial
       analysis for everyone. Remote Sensing of Environment, 202, 18-27.
"""

import functools
from pathlib import Path
from typing import Self

import ee
import numpy as np
import pandas as pd
import platformdirs
import sunwhere
from loguru import logger
from scipy.interpolate import interp1d

from ._base import BaseAtmosphere, build_atmosphere_of_sites
from .conversions import ozone_in_du_to_kg_m2
from ..config import get_option, get_config_path
from ..validation import Latitude, Longitude, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)


def get_database_path() -> Path:
    """Get the path to the MERRA-2 GEE data cache directory.

    Checks for custom directory via config option `merra2_gee.data_dir`.
    If not configured, defaults to platform-specific user data directory.
    Creates the directory if it doesn't exist.

    Returns
    -------
    Path
        Directory path for cached GEE MERRA-2 data files
        
    Examples
    --------
    >>> from pysparta import config
    >>> from pysparta.atmoslib.merra2_geeapi import get_database_path
    >>>
    >>> path = get_database_path()
    >>> print(path)  # ~/.local/share/pysparta/merra2_gee (Linux)
    >>>
    >>> # Configure custom location
    >>> config.set_option('merra2_gee.data_dir', '/data/merra2_gee')
    """
    data_dir = get_option("merra2_gee.data_dir", default=platformdirs.user_data_path("pysparta/merra2_gee"))
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

class MERRA2GEEAtmosphere(
    BaseAtmosphere,
    database_path=get_database_path()
):
    """MERRA-2 atmospheric database via Google Earth Engine.
    
    Provides access to NASA MERRA-2 reanalysis via GEE API. Automatically
    corrects for GEE's latitude grid offset and time stamp convention.
    
    Requires GEE authentication and active project configuration.
    
    See module documentation for setup instructions and examples.
    """

    @classmethod
    def get_filename(cls, year: int, latitude: float, longitude: float) -> Path:
        """Generate cache filename for GEE MERRA-2 data.

        Constructs a standardized filename for cached data with encoded
        coordinates and year.

        Parameters
        ----------
        year : int
            Year for the data
        latitude : float
            Latitude in degrees North [-90, 90]
        longitude : float
            Longitude in degrees East [-180, 180]

        Returns
        -------
        Path
            Absolute path to the .parquet cache file

        Examples
        --------
        >>> path = MERRA2GEEAtmosphere.get_filename(2023, 40.4168, -3.7038)
        >>> print(path.name)
        # "merra2_gee_hourly_404168N_37038W_2023.parquet"
        """
        filename_pattern = "merra2_gee_hourly_{latitude}_{longitude}_{year}.parquet"
        latitude_str = f"{float(latitude)*1e4:.0f}"
        latitude_str = latitude_str[1:]+"S" if latitude_str.startswith("-") else latitude_str + "N"
        longitude_str = f"{float(longitude)*1e4:.0f}"
        longitude_str = longitude_str[1:]+"W" if longitude_str.startswith("-") else longitude_str + "E"
        filename = filename_pattern.format(latitude=latitude_str, longitude=longitude_str, year=year)
        return cls.database_path / filename

    @classmethod
    def at_site(
        cls,
        times: pd.DatetimeIndex,
        latitude: float,
        longitude: float,
        site_name: str | None = None,
    ) -> Self:
        """Retrieve MERRA-2 data from GEE for a specific site.
        
        Downloads data from GEE if not cached, applies spatial/temporal
        corrections, then interpolates to requested times.
        
        Parameters
        ----------
        times : pd.DatetimeIndex
            Time stamps for data retrieval (UTC)
        latitude : float
            Latitude in degrees North [-90, 90]
        longitude : float
            Longitude in degrees East [-180, 180]
        site_name : str, optional
            Name identifier for the site
            
        Returns
        -------
        MERRA2GEEAtmosphere
            Instance with interpolated atmospheric data
            
        Examples
        --------
        >>> import pandas as pd
        >>> from pysparta import config
        >>> config.set_option('merra2_gee.project', 'my-gee-project')
        >>>
        >>> times = pd.date_range("2020-06-15", periods=24, freq="h")
        >>> atm = MERRA2GEEAtmosphere.at_site(
        ...     times=times,
        ...     latitude=36.72,
        ...     longitude=-4.42,
        ...     site_name="Málaga"
        ... )
        >>> result = atm.compute(model="SPARTA")
        
        Notes
        -----
        - Requires `merra2_gee.project` configuration
        - Automatically corrects GEE's 0.25° latitude offset
        - Adjusts time stamps from hour-start to hour-center
        - Data is cached locally to minimize API calls
        """

        latitude = validate_type(latitude, Latitude)
        longitude = validate_type(longitude, Longitude)

        def fetch_and_distill_and_archive(year: int, path: Path) -> None:
            logger.info(f"fetching GEE data for year={int(year)} and path={path.as_posix()}")
            if not (gee_project := get_option("merra2_gee.project")):
                raise ValueError("missing Google cloud's project. Add `project = \"<your_gee_project>\"` "
                                 f"in the `merra2_gee` table in `{get_config_path()}` and reload pysparta "
                                 "or use `pysparta.config.set_option(\'merra2_gee.project\', <your_project>)`")
            logger.info(f"using GEE project <green>{gee_project}</green>")
            data = fetch_merra2_data_from_gee_api(
                latitude=latitude,
                longitude=longitude,
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                project=gee_project)

            data = cls.distill_crude_data(data, latitude, longitude)
            logger.debug(f"{data.head()=}")
            data.to_parquet(path)
            logger.success(f"data downloaded and archived in <blue>{path.name}</blue>")

        # load data from one year before and one year after the requested times_utc, but
        # clipping the years on 2004 and the current year
        paths = []
        years = cls._infer_years_from_times(times)
        for year in sorted(years):
            if not (path := cls.get_filename(year, latitude, longitude)).exists():
                fetch_and_distill_and_archive(year, path)
            paths.append(path)
        logger.debug([path.as_posix() for path in paths])
        data = pd.read_parquet(paths)

        # interpolate to times_utc
        x = times.astype("datetime64[s]").to_numpy().astype(float)
        xi = data.times_utc.astype("datetime64[s]").to_numpy().astype(float)
        data_i = data.drop(columns=["times_utc", "albedo"])
        y = interp1d(xi, data_i.values, kind="quadratic", axis=0)(x)
        y_dict = {key: value for key, value in zip(data_i.columns, y.T)}
        y_dict["albedo"] = interp1d(xi, data.albedo.values, kind="linear", axis=0)(x)
        data_interp = pd.DataFrame({"times": times} | y_dict)

        global_attrs = {
            "title:": "HourlyEarth Engine Data Catalog dataset for SPARTA",
            "source": "NASA/GMAO MERRA-2 reanalysis via Google Earth Engine API",
            "references": "doi:10.5067/KLICLTZ8EM9D, doi:10.5067/Q9QMY5PBNV1T, doi:10.5067/VJAFPLI1CSIV",
        }

        obj = cls()
        obj._atmosphere = build_atmosphere_of_sites(
            times=times,
            latitude=latitude,
            longitude=longitude,
            constituents=data_interp.drop(columns=["times"]),
            site_names=site_name,
            global_attrs=global_attrs)
        return obj

    @staticmethod
    def _infer_years_from_times(times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex) -> list[int]:
        """Infer which years are needed based on requested times.
        
        Pads with extra years if times are within 3 hours of year boundaries
        to ensure smooth temporal interpolation.
        
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
        >>> times = pd.date_range("2020-06-01", "2020-08-31", freq="h")
        >>> years = MERRA2GEEAtmosphere._infer_years_from_times(times)
        >>> print(years)  # [2020]
        
        >>> # Near year boundary - includes padding
        >>> times = pd.date_range("2020-01-01 00:00", "2020-01-01 02:00", freq="h")
        >>> years = MERRA2GEEAtmosphere._infer_years_from_times(times)
        >>> print(years)  # [2019, 2020]
        """
        the_times = times if isinstance(times, pd.DatetimeIndex) else pd.to_datetime(times)
        years = set(the_times.year)
        if (the_times[0] - pd.to_datetime(f"{min(years)}-01-01 00:00:00")) < pd.Timedelta(3, "h"):
            years.add(min(years)-1)
        if (pd.to_datetime(f"{max(years)+1}-01-01 00:00:00") - the_times[-1]) < pd.Timedelta(3, "h"):
            years.add(max(years)+1)
        return sorted(years)

    @staticmethod
    def distill_crude_data(data: pd.DataFrame, lat: Latitude, lon: Longitude) -> pd.DataFrame:
        """Refine raw GEE MERRA-2 data for clear-sky modeling.
        
        Performs post-processing on GEE data:
        1. Adjusts time stamps from hour-start to hour-center (NASA convention)
        2. Calculates solar zenith angle for albedo masking
        3. Computes Ångström turbidity coefficient (beta) from AOD and alpha
        4. Calculates aerosol single-scattering albedo (SSA)
        5. Converts ozone from DU to kg/m²
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data from GEE API with columns: TOTEXTTAU, TOTSCATAU,
            TOTANGSTR, PS, TO3, TQV, ALBEDO
        lat : float
            Site latitude (for solar position calculation)
        lon : float
            Site longitude (for solar position calculation)
            
        Returns
        -------
        pd.DataFrame
            Processed DataFrame with columns: times_utc, albedo, pressure,
            ozone, pwater, beta, alpha, ssa
            
        Notes
        -----
        GEE time stamps are at hour start (e.g., 01:00 UTC), but MERRA-2
        hourly averages represent the period centered at half-past (e.g., 01:30 UTC).
        This function adds 30 minutes to correct the convention.
        
        Albedo is masked to 0 for solar zenith angles > 89°.
        """

        # N.B. The Half-Hour Shift (Time-Averaged vs. GEE Timestamp)
        # These MERRA-2 products are time-averaged (tavg1_2d) hourly collections.
        #  - *The NASA Convention*: In raw MERRA-2 NetCDF files, hourly averages represent the mean
        #    value across a full hour. NASA time-stamps these intervals at the midpoint of the hour.
        #    For example, the average for the hour between 01:00 UTC and 02:00 UTC is stamped
        #    as 01:30 UTC.
        #  - *The GEE Adjustment (system:time_start)*: Google Earth Engine forces all dataset tracking
        #    into discrete start/end boundaries. To make this compatible with standard data querying,
        #    GEE strips the half-hour offset from the tracking metadata. The system:time_start property
        #    is adjusted back to the beginning of that hourly window (e.g., 01:00:00 UTC).

        # hence, make a tz-naive datetimeindex and move the timestamps to the center of the interval...
        data = (data
                .set_index("times_utc", drop=True)
                .tz_convert(tz=None)
                .pipe(lambda df: df.set_index(df.index + pd.Timedelta(30, "min"))))

        solpos = sunwhere.sites(data.index, lat, lon)
        sza = solpos.sza.isel(location=0).to_pandas()

        return (
            data.assign(
                albedo=data.ALBEDO.where(sza < 89., 0.),
                pressure=data.PS,  # Pa
                ozone=ozone_in_du_to_kg_m2(data.TO3),  # DU to kg m-2
                pwater=data.TQV,  # kg m-2
                beta=data.TOTEXTTAU*(0.55**data.TOTANGSTR),  # Angstrom's turbidity
                alpha=data.TOTANGSTR,  # Angstrom's wavelength parameter
                ssa=(data.TOTSCATAU/data.TOTEXTTAU).clip(0., 1.))
            .drop(columns=["TOTEXTTAU", "TOTSCATAU", "TOTANGSTR", "PS", "TO3", "TQV", "ALBEDO"])
            .rename_axis("times_utc", axis=0)
            .interpolate()  # to fill nans (sometimes albedo has remaining nans)
            .reset_index()
        )

@functools.lru_cache()
def fetch_merra2_data_from_gee_api(
    latitude: Latitude,
    longitude: Longitude,
    start_date: str,  # YYYY-mm-dd
    end_date: str,  # YYYY-mm-dd
    project: str,  # = "series-temporales-merra2",
) -> pd.DataFrame:
    """Fetch MERRA-2 data from Google Earth Engine.
    
    Retrieves atmospheric constituent data from three GEE MERRA-2 collections,
    applying the necessary latitude correction for GEE's grid offset.
    Results are cached via LRU cache.
    
    Parameters
    ----------
    latitude : float
        Latitude in degrees North [-90, 90]
    longitude : float
        Longitude in degrees East [-180, 180]
    start_date : str
        Start date in format "YYYY-MM-DD"
    end_date : str
        End date in format "YYYY-MM-DD"
    project : str
        Google Earth Engine project ID
        
    Returns
    -------
    pd.DataFrame
        Raw data with columns from all three collections:
        - TOTEXTTAU, TOTSCATAU, TOTANGSTR (aerosol collection)
        - PS, TO3, TQV (single-level collection)
        - ALBEDO (radiation collection)
        
    Examples
    --------
    >>> from pysparta.atmoslib.merra2_geeapi import fetch_merra2_data_from_gee_api
    >>> data = fetch_merra2_data_from_gee_api(
    ...     latitude=36.72,
    ...     longitude=-4.42,
    ...     start_date="2020-06-01",
    ...     end_date="2020-06-30",
    ...     project="my-gee-project"
    ... )
    >>> print(data.columns)
    
    Notes
    -----
    Critical spatial correction:
    GEE's MERRA-2 ingestion shifted the latitude grid by 0.25° northward.
    This function automatically subtracts 0.25° from the query latitude
    to retrieve the correct NASA grid cell.
    
    Collections accessed:
    - NASA/GSFC/MERRA/aer/2: Aerosol optical properties
    - NASA/GSFC/MERRA/slv/2: Single-level diagnostics  
    - NASA/GSFC/MERRA/rad/2: Radiation parameters
    
    Raises
    ------
    ee.EEException
        If GEE authentication fails or project is invalid
    """

    latitude = validate_type(latitude, Latitude)
    longitude = validate_type(longitude, Longitude)
    start_date = f"{pd.Timestamp(start_date).date():%Y-%m-%d}"  # validate date
    end_date = f"{pd.Timestamp(end_date).date():%Y-%m-%d}"  # validate date

    ee.Authenticate()
    logger.debug("ee authenticated")

    ee.Initialize(project=project)
    logger.debug(f"ee initialized: {project=}")

    # N.B. Grid Inconsistency
    # The native NASA MERRA-2 model is computed on a global regular grid with an asymmetric resolution
    # of 0.5° in latitude by 0.625° in longitude.The problem stems from how the coordinates of the pixel
    # centers are indexed:
    #  - *In NASA (Original Format)*: Latitude cells are centered at the poles and the equator, stepping
    #    by 0.5° starting exactly at -90.0° (e.g., -90.0, -89.5, ..., 0.0, 0.5, ..., 90.0).
    #  - *In Google Earth Engine*: During the automated bulk ingestion pipeline of the raster data, Google's
    #    system interpreted or registered the spatial origin by aligning the pixel edge or applying a half-cell
    #    resolution offset on the Y-axis. By interpreting the corner instead of the center, the entire MERRA-2
    #    matrix in GEE ended up shifted vertically northward by exactly 0.25° (half of the 0.5° resolution).
    # To fix of the problem, counteracting GEE's ingestion error and forcing the API to select the correct
    # physical NASA cell, the user must manually subtract 0.25° from the target latitude before generating the
    # ee.Geometry.Point object. By artificially pulling the query coordinate a quarter-degree southward, one
    # compensates for Google's raster misalignment, ensuring that getRegion() reads the actual pixel that NASA
    # calculated for the site's true physical location.
    latitude_gee_corr = latitude - 0.25

    # define site...
    point = ee.Geometry.Point([longitude, latitude_gee_corr])
    logger.debug(f"{point=}")

    collections_and_variables = {
        "NASA/GSFC/MERRA/aer/2": ["TOTEXTTAU", "TOTSCATAU", "TOTANGSTR"],
        "NASA/GSFC/MERRA/slv/2": ["PS", "TO3", "TQV"],
        "NASA/GSFC/MERRA/rad/2": ["ALBEDO"],
    }

    def fetch_collection(collection: str) -> pd.DataFrame:
        variables = collections_and_variables[collection]
        logger.debug(f"fetching collection `{collection}` ({variables=})")
        img_coll = ee.ImageCollection(collection).filterDate(start_date, end_date).select(variables)
        raw_data = img_coll.getRegion(point, scale=50000).getInfo()  # scale chosen for the resolution of MERRA-2
        df_local = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        df_local["date"] = pd.to_datetime(df_local["time"], unit="ms", utc=True)
        df_local = (
            df_local
            .set_index("date", drop=True)
            .rename_axis("times_utc")
            .sort_index(axis=0)
            .drop(columns=["id", "latitude", "longitude", "time"], errors="ignore"))
        logger.debug(f"fetched collection `{collection}` ({variables=})")
        return df_local

    df = pd.concat([fetch_collection(coll) for coll in collections_and_variables], axis=1)
    logger.success(f"fetched MERRA-2 data from {start_date} to {end_date} at "
                   f"(lon, lat) = ({longitude}, {latitude})")
    return df.reset_index()
