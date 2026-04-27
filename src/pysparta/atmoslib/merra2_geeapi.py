
import functools
from pathlib import Path
from typing import Self

import ee
import numpy as np
import pandas as pd
import platformdirs
import sunwhere
import xarray as xr
from loguru import logger
from scipy.interpolate import interp1d

from ._base import BaseAtmosphere, make_cf_compliant
from .conversions import ozone_in_du_to_kg_m2
from ..config import get_option, get_config_path
from ..validation import Latitude, Longitude, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)


def get_database_path() -> Path:
    """Get the path to the archive storage directory.

    This function first checks for a custom directory path provided via system 
    options (`crs_soda.data_dir`). If no custom path is configured, it defaults to the
    platform-specific user data directory (typically, ~/.config in linux flavors).
    In both cases, the function ensures the directory actually exists on the
    filesystem before returning.

    Returns:
        Path: The filesystem path to the data directory (either the custom 
            configured directory or the system default).
    """
    data_dir = get_option("merra2_gee.data_dir", default=platformdirs.user_data_path("pysparta/merra2_gee"))
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

class MERRA2GEEAtmosphere(
    BaseAtmosphere,
    database_path=get_database_path()
):

    @classmethod
    def get_filename(cls, year: int, latitude: float, longitude: float) -> Path:
        r"""Generates the full path for a specific archive file.

        Constructs a filename using a standardized pattern for SoDA McClear data, 
        including versioning, encoded coordinates, and year. The coordinates are 
        multiplied by \(10^4\) and suffixed with cardinal direction indicators.

        Args:
            year: The reference year for the data.
            latitude: Geographical latitude, in decimal degrees (-90, 90).
            longitude: Geographical longitude, in decimal degrees [-180, 180).
            version: Version string for the dataset (e.g., "0.1.0").

        Returns:
            Path: The absolute path to the .parquet file within the archive location.

        Examples:
            >>> get_archive_filename(2023, 40.4168, -3.7038, "1.0.0")
            # Resulting filename: "crs_soda_mcclear_v1.0.0_404168N_37038W_2023.parquet"

        Notes:
            Coordinates are formatted as follows:
            - Multiplied by \(10^4\) and converted to integer strings.
            - Signs are replaced by suffixes: 'N'/'S' for latitude and 'E'/'W' for longitude.
            - Example: A latitude of `-12.34` becomes `123400S`.
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

        # dimensions...
        dims = ("time", "site")

        # coordinates...
        coords = {"time": ("time", times), "lat": ("site", [latitude]), "lon": ("site", [longitude])}
        if site_name is not None:
            coords.update({"site_name": ("site", [site_name])})

        # variables...
        data_vars = {variable: (dims, data_interp[variable].values[:, None])
                     for variable in data.columns.drop("times_utc")}

        interp_dataset = xr.Dataset(data_vars, coords)

        obj = cls()
        obj._atmosphere = make_cf_compliant(interp_dataset, overwrite=True)
        return obj

    @staticmethod
    def _infer_years_from_times(times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex) -> list[int]:
        the_times = times if isinstance(times, pd.DatetimeIndex) else pd.to_datetime(times)
        years = set(the_times.year)
        if (the_times[0] - pd.to_datetime(f"{min(years)}-01-01 00:00:00")) < pd.Timedelta(3, "h"):
            years.add(min(years)-1)
        if (pd.to_datetime(f"{max(years)+1}-01-01 00:00:00") - the_times[-1]) < pd.Timedelta(3, "h"):
            years.add(max(years)+1)
        return sorted(years)

    @staticmethod
    def distill_crude_data(data: pd.DataFrame, lat: Latitude, lon: Longitude) -> pd.DataFrame:

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
