
import functools
import re
from io import StringIO
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import platformdirs
import requests
import xarray as xr
from loguru import logger
from scipy.interpolate import interp1d

from ._base import BaseAtmosphere, make_cf_compliant
from .conversions import ozone_in_du_to_kg_m2
from ..config import get_option, get_config_path
from ..validation import Latitude, Longitude, SodaStream, SodaTimeStep, validate_type

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
    data_dir = get_option("crs_soda.data_dir", default=platformdirs.user_data_path("pysparta/crs_soda"))
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class CRSSODAAtmosphere(
    BaseAtmosphere,
    database_path=get_database_path()
):

    @classmethod
    def get_filename(cls, year: int, latitude: float, longitude: float, version: str) -> Path:
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
        filename_pattern = "crs_soda_mcclear_v{version}_{latitude}_{longitude}_{year}.parquet"
        latitude_str = f"{float(latitude)*1e4:.0f}"
        latitude_str = latitude_str[1:]+"S" if latitude_str.startswith("-") else latitude_str + "N"
        longitude_str = f"{float(longitude)*1e4:.0f}"
        longitude_str = longitude_str[1:]+"W" if longitude_str.startswith("-") else longitude_str + "E"
        filename = filename_pattern.format(version=version, latitude=latitude_str, longitude=longitude_str, year=year)
        return cls.database_path / filename

    @classmethod
    def at_site(
        cls,
        times: pd.DatetimeIndex,
        latitude: float,
        longitude: float,
        site_name: str | None = None,
    ) -> Self:

        version: str = "1.0.0"
        save_csv: bool = True  # for debugging and reproducibility. The csv files are saved in the same directory as the parquet files, with the same name but .csv extension instead of .parquet

        latitude = validate_type(latitude, Latitude)
        longitude = validate_type(longitude, Longitude)

        def fetch_and_distill_and_archive(year: int, path: Path) -> None:
            if (user_email := get_option("crs_soda.user_email")) is None:
                raise ValueError("missing soda user. Add `user_email = \"<your_email_for_crs_soda>\"` in "
                                 f"the `crs_soda` table in `{get_config_path()}` and reload pysparta or use "
                                 "`pysparta.config.set_option(\'crs_soda.user_email\', <your_email_for_crs_soda>)`")
            data, metadata = fetch_crs_data_from_soda_api(
                latitude=latitude,
                longitude=longitude,
                date_begin=f"{year}-01-01",
                date_end=f"{year}-12-31",
                user_email=user_email,
                time_step="PT01M",
                stream="mcclear",
                version=version,
                timeout=30,
                to_csv=path.with_suffix(".csv") if save_csv else None)  # save the raw response as csv for debugging and reproducibility

            data = cls.distill_crude_data(data, metadata)
            logger.debug(f"{data.head()=}")
            data.to_parquet(path)
            logger.success(f"data downloaded and archived: <blue>{path.name}</blue>")

        # load data from one year before and one year after the requested times_utc, but
        # clipping the years on 2004 and the current year
        paths = []
        years = cls._infer_years_from_times(times)
        for year in sorted(years):
            if not (path := cls.get_filename(year, latitude, longitude, version)).exists():
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
    def distill_crude_data(data: pd.DataFrame, metadata: list[str]) -> pd.DataFrame:
        r"""Refines and enriches raw CRS data for clear-sky evaluations.

        This function performs several post-processing steps:
        1.  Calculates AOD550 and estimates the Angström exponent (\(\alpha\)) 
            using a weighted average of aerosol mixtures if missing.
        2.  Estimates local barometric pressure from altitude using Laplace's formula.
        3.  Resamples the dataset to 1-hour intervals with center-time alignment.

        Args:
            data: Raw DataFrame retrieved from the SoDA API.
            metadata: List of metadata strings containing site information (e.g., altitude).

        Returns:
            pd.DataFrame: A cleaned DataFrame containing columns: `times_utc`, 
                `pressure`, `albedo`, `pwater`, `ozone`, `aod550`, `beta`, and `alpha`.

        Notes:
            The \(\alpha\) estimation uses typical values for aerosol species 
            (DU, SS, BC, etc.) based on Bozzo et al. (2017).

        References:
            - Bozzo et al. (2017). Implementation of a CAMS-based aerosol climatology 
            in the IFS, ECMWF. [Online](https://www.ecmwf.int/sites/default/files/elibrary/2017/17219-implementation-cams-based-aerosol-climatology-ifs.pdf)
        """
        def barometric_formula_laplace(
            z: float,
            L: float = 0.0065,  # vertical thermal gradient, K m-1
            Po: float = 101325.,  # sea-level pressure, Pa
            To: float = 288.15,  # sea-level temperature, K
        ) -> float:

            g = 9.80665  # acceleration of gravity, m s-2
            M = 0.02896  # dry air molar mass, kg mol-1
            R = 8.31447  # state constant of ideal gas, J mol-1 K-1

            gamma = (g*M) / (R*L)
            return Po * (1.-(L*z)/To)**gamma

        # The cams radiation service does not provide alpha. I make then a rough estimation
        # from typical alpha values of each aerosol species. For a better estimation, I could
        # have used the description of the aerosol mixtures (DU, BC, OM...) used in the
        # CAMS Reanalysis EC4, which is provided in Bozzo et al. (2017) [1], combined
        # with the OPAC's characterization of the spectral properties and the distribution
        # of single species and mixtures, but this requires too much effort for the moment.
        # [1] Bozzo et al. (2017) Implementation of a CAMS-based aerosol climatology in
        #     the IFS, ECMWF. Available at: https://www.ecmwf.int/sites/default/files/elibrary/2017/17219-implementation-cams-based-aerosol-climatology-ifs.pdf
        alpha_of_mixtures = {
            "DU": 0.3,  # desert dust. Large particles. CAMS uses 3-5 bins. Coarser particles have alpha close to 0
            "SS": 0.5,  # sea salt. Large particles from sea. Highly hygroscopic. Humidity affects their size
            "BC": 1.2,  # black carbon. Small "dark" particles from combustion processes. Sharp spectral dependence.
            "OR": 1.8,  # organic matter. Fine particles. Includes primary and secondary organic matters.
            "SU": 1.7,  # sulphates. Very fine particles, purely scatterers. Sharp spectral dependence.
            "NI": 1.9,  # nitrates. Similar to sulphates, but mostly in the fine mode.
            "AM": 1.9,  # ammonium. Similar to sulphates, but mostly in the fine mode.
        }

        data = data.assign(
            times_utc=pd.to_datetime(data["Observation period"].str.split("/").str[0]),
            aod550=data.filter(like="AOD").sum(axis=1),
            # N.B. The sza column in the crs data for night-time is 0 !!!
            albedo=data.albedo.where((data.sza < 89) & (data.sza != 0.), 0))

        # make a weighted average of the alphas of the mixtures
        weighted_alpha = 0.
        existing_aods = []
        for mixture, alpha in alpha_of_mixtures.items():
            if (aod_label := f"AOD {mixture}") in data:
                weighted_alpha += alpha *data[aod_label]
                existing_aods.append(aod_label)
        if len(existing_aods):
            weighted_alpha /= data.get(existing_aods).sum(axis=1)

        data["alpha"] = data["alpha"].where(data["alpha"].notna(), weighted_alpha)
        data["beta"] = data["aod550"]*(0.55**data["alpha"])

        # estimate surface pressure from altitude (surface pressure is missing in crs)
        regex = re.compile(r"^#\s*Altitude \(m\):\s*([-+]?\d*\.?\d+)")
        for line in metadata:
            if (m := regex.search(line)):
                altitude = float(m.groups()[0])
                break

        data["pressure"] = barometric_formula_laplace(altitude)  # surface pressure, Pa
        data["ozone"] = ozone_in_du_to_kg_m2(data["tco3"])  # DU to kg m-2
        data["pwater"] = data["tcwv"]  # in kg m-2

        # keep only the relevant columns for clear-sky evaluations with sparta
        data = data.get(["times_utc", "pressure", "albedo", "pwater", "ozone", "aod550", "beta", "alpha"])

        # resample hourly: the 1-min dataset is just an interpolation. It does not
        # make sense wasting disk space just to store this kind of 1-min data
        return (data.set_index("times_utc")
                .resample("1h").mean()  # time alignment = left
                .pipe(lambda df: df.set_index(df.index + pd.Timedelta("30min")))  # time alignment = center
                .reset_index())

@functools.lru_cache()
def fetch_crs_data_from_soda_api(
    latitude: Latitude,
    longitude: Longitude,
    date_begin: str,
    date_end: str,
    user_email: str,
    time_step: SodaTimeStep,
    stream: SodaStream,
    version: str = "1.0.0",
    timeout: int = 30,
    to_csv: Path | str = None,
) -> tuple[pd.DataFrame, list[str]]:
    r"""Retrieves radiation and atmospheric data from the SoDA CRS WPS service.

    This function performs a synchronous GET request to the SoDA Solar Data API,
    parsing the returned CSV-like format into a structured DataFrame and a list
    of metadata strings. Results are cached to minimize redundant API calls.

    Args:
        latitude: Geographical latitude, in decimal degrees (-90, 90).
        longitude: Geographical longitude, in decimal degrees [-180, 180).
        date_begin: Start date of the period, YYYY-MM-DD.
        date_end: End date of the period, YYYY-MM-DD.
        user_email: Registered email for the SoDA service. If not provided, 
            the system searches in the configuration file (`soda_user_email`).
        time_step: Temporal resolution of the data. See [SodaTimeStep][pysparta.types.SodaTimeStep]
            for allowed values.
        stream: The specific SoDA data stream to query. See [SodaStream][pysparta.types.SodaStream]
            for allowed values.
        version: WPS service version. Defaults to "1.0.0".
        timeout: Request timeout in seconds. Defaults to 30.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:
            - The processed DataFrame with CRS data.
            - A list of metadata lines (comments starting with '#').

    Raises:
        requests.exceptions.HTTPError: If the API returns an error status code,
            including the specific SoDA exception text in the reason.
    """
    latitude = validate_type(latitude, Latitude)
    longitude = validate_type(longitude, Longitude)
    summarization = validate_type(time_step, SodaTimeStep)
    stream = validate_type(stream, SodaStream)

    date_begin = f"{pd.Timestamp(date_begin):%Y-%m-%d}"
    date_end = f"{pd.Timestamp(date_end):%Y-%m-%d}"

    data_inputs = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "date_begin": date_begin,
        "date_end": date_end,
        "altitude": f"{-999:d}",
        "summarization": summarization,
        "time_ref": "UT",
        "verbose": "true" if summarization.casefold() == "pt01m" else "false",
        "username": user_email.replace("@", "%2540")}
    data_inputs = ";".join([key + "=" + value for key, value in data_inputs.items()])

    parameters = {
        "Service": "WPS",
        "Request": "Execute",
        "Identifier": f"get_{stream.lower()}",
        "version": version,
        "RawDataOutput": "irradiation"
    }

    base_url = "https://api.soda-solardata.com/service/wps"
    url = base_url + "?DataInputs=" + data_inputs
    logger.debug(f"{url=}")
    res = requests.get(url, params=parameters, timeout=timeout)

    if res.ok:
        logger.debug("Response Ok!")
        data_str = res.content.decode("utf-8")
        if to_csv is not None:
            with open(to_csv, "w") as f:
                f.write(data_str)
            logger.info(f"raw response saved to <blue>{to_csv}</blue>")
        metadata_lines = []
        for line in data_str.splitlines():
            if line.strip().startswith("#"):
                metadata_lines.append(line)
                continue
            break
        header = metadata_lines[-1].strip().strip("#").strip().split(";")
        logger.debug(f"{header=}")
        metadata = metadata_lines[:-1]
        data = pd.read_csv(StringIO(data_str), comment="#", sep=";").set_axis(header, axis=1)
        logger.debug(f"{data.shape=}\n{data.describe()}")
        return data, metadata
    else:
        errors = res.text.split("ows:ExceptionText")[1][1:-2]
        res.reason = "%s: <%s>" % (res.reason, errors)
        res.raise_for_status()
