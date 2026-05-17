"""MERRA-2 Daily Atmospheric Data Interface.

This module provides access to MERRA-2 (Modern-Era Retrospective analysis for 
Research and Applications, Version 2) daily atmospheric data. The data includes
aerosol optical properties, precipitable water, ozone, surface pressure, and
albedo at daily temporal resolution.

The MERRA2DailyAtmosphere class enables:
    - Extraction at specific geographic locations (sites)
    - Extraction on regular lat/lon grids
    - Temporal and spatial interpolation
    - Automatic data caching and loading

Data Source:
    MERRA-2 daily data is stored in Zarr format and organized by year.
    Data can be cached locally or downloaded on demand.

Examples:
    >>> import pandas as pd
    >>> from spartasolar.atmoslib import MERRA2DailyAtmosphere
    
    >>> # Load atmospheric data for a specific location
    >>> times = pd.date_range("2020-01-01", periods=10, freq="D")
    >>> atmos = MERRA2DailyAtmosphere.at_sites(
    ...     times=times,
    ...     latitude=40.4168,  # Madrid
    ...     longitude=-3.7038,
    ...     site_names="Madrid"
    ... )
    
    >>> # Load data on a regular grid
    >>> lats = [35, 40, 45]
    >>> lons = [-10, -5, 0, 5]
    >>> atmos = MERRA2DailyAtmosphere.on_regular_grid(
    ...     times=times,
    ...     latitude=lats,
    ...     longitude=lons
    ... )
    
    >>> # Access atmospheric variables
    >>> pressure = atmos.get("pressure")
    >>> albedo = atmos.get("albedo")

See Also:
    - MERRA2LTAAtmosphere: Long-term average MERRA-2 climatology
    - BaseAtmosphere: Base class defining the atmosphere interface
"""

from pathlib import Path
from typing import Self, Sequence

import huggingface_hub as Hf
import numpy as np
import numpy.typing as npt
import pandas as pd
import platformdirs
import xarray as xr
from loguru import logger

from ._base import BaseAtmosphere, build_atmosphere_of_sites, build_atmosphere_on_regular_grid
from ..config import get_option
from ..validation import Latitude, Longitude, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)


def get_database_path() -> Path:
    """Get the local path to MERRA-2 daily database.
    
    Returns the configured data directory path or the default platform-specific
    user data path. Creates the directory if it doesn't exist.
    
    Returns:
        Path: Directory path where MERRA-2 daily data is stored/cached.
        
    Examples:
        >>> from spartasolar.atmoslib.merra2_daily import get_database_path
        >>> db_path = get_database_path()
        >>> print(db_path.exists())
        True
    """
    user_path = (get_option("merra2_daily.data_dir") or
                 platformdirs.user_data_path('spartasolar/merra2-daily'))
    if not user_path.exists():
        user_path.mkdir(parents=True, exist_ok=True)
    return user_path


class MERRA2DailyAtmosphere(
    BaseAtmosphere,
    database_path=get_database_path()
):
    """MERRA-2 daily atmospheric data accessor.
    
    Provides methods to load and interpolate MERRA-2 daily atmospheric data
    for specific locations or regular grids. Data is automatically cached
    locally and loaded from Zarr archives organized by year.
    
    The class inherits from BaseAtmosphere and provides two main factory methods:
        - at_sites(): Extract data at specific point locations
        - on_regular_grid(): Extract data on a regular lat/lon mesh
    
    Attributes:
        database_path: Path to local MERRA-2 data storage directory
        
    Available Variables:
        - pressure: Surface air pressure [Pa]
        - albedo: Surface albedo [0-1]
        - pwater: Precipitable water [kg/m²]
        - ozone: Total column ozone [kg/m²]
        - beta: Angström turbidity parameter
        - alpha: Angström wavelength exponent
        - ssa: Aerosol single scattering albedo
    """

    @classmethod
    def at_sites(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float] | float,
        longitude: Sequence[float] | float,
        site_names: Sequence[str] | None = None,
    ) -> Self:
        """Load atmospheric data at specific geographic locations.
        
        Extracts and interpolates MERRA-2 data for one or more point locations.
        Performs bilinear spatial interpolation and quadratic temporal interpolation.
        
        Args:
            times: Time points for data extraction. Can be numpy datetime64 array
                or pandas DatetimeIndex.
            latitude: Latitude coordinate(s) in decimal degrees. Single value or
                sequence. Range: -90° < lat < 90°.
            longitude: Longitude coordinate(s) in decimal degrees. Single value or
                sequence. Range: -180° ≤ lon < 180°.
            site_names: Optional names for the sites. If provided, added as a
                coordinate in the output dataset.
                
        Returns:
            MERRA2DailyAtmosphere: Instance containing interpolated atmospheric data.
            
        Raises:
            ValueError: If latitude and longitude have different lengths, or if
                coordinates are out of valid range.
            NotImplementedError: If required data files are not found locally
                and downloading is not yet implemented.
                
        Examples:
            >>> import pandas as pd
            >>> from spartasolar.atmoslib import MERRA2DailyAtmosphere
            
            >>> # Single location
            >>> times = pd.date_range("2020-01-01", periods=5, freq="D")
            >>> atmos = MERRA2DailyAtmosphere.at_sites(
            ...     times=times,
            ...     latitude=40.4168,
            ...     longitude=-3.7038,
            ...     site_names="Madrid"
            ... )
            
            >>> # Multiple locations
            >>> lats = [40.4168, 41.3851, 36.7213]  # Madrid, Barcelona, Málaga
            >>> lons = [-3.7038, 2.1734, -4.4214]
            >>> names = ["Madrid", "Barcelona", "Málaga"]
            >>> atmos = MERRA2DailyAtmosphere.at_sites(
            ...     times=times,
            ...     latitude=lats,
            ...     longitude=lons,
            ...     site_names=names
            ... )
            
            >>> # Access data
            >>> pressure = atmos.get("pressure")
            >>> print(pressure.dims)
            ('time', 'site')
        """

        latitude = [latitude] if isinstance(latitude, (float, int)) else latitude
        latitude = np.asarray([validate_type(lat, Latitude) for lat in latitude], dtype=float).reshape(-1)
        longitude = [longitude] if isinstance(longitude, (float, int)) else longitude
        longitude = np.asarray([validate_type(lon, Longitude) for lon in longitude], dtype=float).reshape(-1)

        if len(latitude) != len(longitude):
            raise ValueError('latitude and longitude must have the same length')

        # load the dataset. Check for local availability. If not available, download.
        dataset = cls._load_dataset(times)

        # lat-lon interpolation
        output_lat = xr.DataArray(latitude, dims="site", name="lat")
        output_lon = xr.DataArray(longitude, dims="site", name="lon")
        # # if interpolation was `nearest` it is faster to use .sel than .interp
        # output_dataset = dataset.sel(lat=output_lat, lon=output_lon, method='nearest')
        # output_dataset = dataset.interp(lat=output_lat, lon=output_lon, method='nearest')
        output_dataset = dataset.interp(lat=output_lat, lon=output_lon, method='linear')

        # time interpolation
        if "time" in output_dataset.coords:
            output_dataset = output_dataset.interp(time=times, method='quadratic')

        global_attrs = {
            "title": "Daily Clear-sky Atmospheric Dataset for SPARTA",
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
        latitude: Sequence[float] | float,
        longitude: Sequence[float] | float,
    ) -> Self:
        """Load atmospheric data on a regular lat/lon grid.
        
        Extracts and interpolates MERRA-2 data onto a user-specified regular grid.
        The output has dimensions (time, lat, lon). NaN values in albedo are filled
        with zeros.
        
        Args:
            times: Time points for data extraction.
            latitude: Latitude coordinates for the grid in decimal degrees.
                Must be a sequence (list, array).
            longitude: Longitude coordinates for the grid in decimal degrees.
                Must be a sequence (list, array).
                
        Returns:
            MERRA2DailyAtmosphere: Instance containing gridded atmospheric data.
            
        Raises:
            ValueError: If coordinates are out of valid range.
            NotImplementedError: If required data files are not found locally.
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from spartasolar.atmoslib import MERRA2DailyAtmosphere
            
            >>> # Define a regular grid over Iberian Peninsula
            >>> times = pd.date_range("2020-06-01", periods=10, freq="D")
            >>> lats = np.linspace(36, 44, 9)  # 9 latitude points
            >>> lons = np.linspace(-10, 4, 15)  # 15 longitude points
            
            >>> atmos = MERRA2DailyAtmosphere.on_regular_grid(
            ...     times=times,
            ...     latitude=lats,
            ...     longitude=lons
            ... )
            
            >>> # Access gridded data
            >>> albedo = atmos.get("albedo")
            >>> print(albedo.dims)
            ('time', 'lat', 'lon')
            >>> print(albedo.shape)
            (10, 9, 15)
        """

        latitude = [latitude] if isinstance(latitude, (float, int)) else latitude
        latitude = np.asarray([validate_type(lat, Latitude) for lat in latitude], dtype=float).reshape(-1)
        longitude = [longitude] if isinstance(longitude, (float, int)) else longitude
        longitude = np.asarray([validate_type(lon, Longitude) for lon in longitude], dtype=float).reshape(-1)

        # load the dataset. Check for local availability. If not available, download.
        dataset = cls._load_dataset(times)
        dataset["albedo"] = dataset["albedo"].fillna(0.)

        # lat-lon interpolation
        output_dataset = dataset.interp(lat=latitude, lon=longitude, method='linear')

        # time interpolation
        if "time" in output_dataset.coords:
            output_dataset = output_dataset.interp(time=times, method='quadratic')

        global_attrs = {
            "title": "Daily Clear-sky Atmospheric Dataset for SPARTA",
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
        """Infer which years of data are needed for temporal interpolation.
        
        Determines the set of years required to cover the requested time range,
        including padding years if times are close to year boundaries (within 3 days).
        This ensures smooth temporal interpolation at year transitions.
        
        Args:
            times: Array of datetime values for which data is needed.
            
        Returns:
            list[int]: Sorted list of years whose data files should be loaded.
            
        Examples:
            >>> import pandas as pd
            >>> from spartasolar.atmoslib.merra2_daily import MERRA2DailyAtmosphere
            
            >>> # Mid-year dates - only 2020
            >>> times = pd.date_range("2020-06-01", "2020-07-01", freq="D")
            >>> MERRA2DailyAtmosphere._infer_years_from_times(times)
            [2020]
            
            >>> # Spanning years - both 2020 and 2021
            >>> times = pd.date_range("2020-12-01", "2021-01-31", freq="D")
            >>> MERRA2DailyAtmosphere._infer_years_from_times(times)
            [2020, 2021]
            
            >>> # Near year start - includes previous year for interpolation
            >>> times = pd.date_range("2020-01-01", "2020-01-02", freq="D")
            >>> MERRA2DailyAtmosphere._infer_years_from_times(times)
            [2019, 2020]
            
            >>> # Near year end - includes next year for interpolation
            >>> times = pd.date_range("2020-12-30", "2020-12-31", freq="D")
            >>> MERRA2DailyAtmosphere._infer_years_from_times(times)
            [2020, 2021]
        """
        the_times = times if isinstance(times, pd.DatetimeIndex) else pd.to_datetime(times)
        years = set(the_times.year)
        if (the_times[0] - pd.to_datetime(f"{min(years)}-01-01 12")) < pd.Timedelta(3, "D"):
            years.add(min(years)-1)
        if (pd.to_datetime(f"{max(years)}-12-31 12") - the_times[-1]) < pd.Timedelta(3, "D"):
            years.add(max(years)+1)
        return sorted(years)

    @classmethod
    def _ensure_all_paths_are_local(cls, paths: list[Path]) -> None:

        START_YEAR = 1999
        END_YEAR = 2018

        def missing_paths():
            for path in filter(lambda p: not p.exists(), paths):
                year = int(path.stem)
                if not (START_YEAR <= year <= END_YEAR):
                    raise ValueError(f"requested year {year} is out of range for "
                                     f"MERRA-2 daily data ({START_YEAR}-{END_YEAR})")
                yield path, year

        def download_Hf_yearly_chunk(year: int):
            logger.info(f"downloading MERRA-2 daily data for year {year} from Hugging Face Hub...")
            Hf.snapshot_download(
                repo_id="josearuizarias/merra2-daily-clearsky",
                repo_type="dataset",
                allow_patterns=[f"{year}/*"],
                local_dir=cls.database_path)

        for path, year in missing_paths():
            download_Hf_yearly_chunk(year)

    @classmethod
    def _load_dataset(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
    ) -> xr.Dataset:

        years = cls._infer_years_from_times(times)
        paths = [cls.database_path / str(year) for year in sorted(years)]
        cls._ensure_all_paths_are_local(paths)  # if any path is missing, download it

        logger.debug(f"loading paths: {paths}")

        # para abrir múltiples zarr anuales concatenados
        return xr.open_mfdataset(
            sorted(paths),  # lista de ficheros ordenados cronológicamente
            engine="zarr",
            chunks={},  # respeta los chunks originales (los físicos)
            # combine='nested' vs 'by_coords':
            #  - Si tus archivos ya tienen una coordenada de tiempo correcta, usa by_coords.
            #  - Si quieres forzar el orden usa nested y especifica concat_dim (pero entonces yo debo
            #    ordenar los paths cronológicamente)
            combine="nested",   # 'nested' es más rápido si conoces el orden (y lo conozco de hecho!)
            concat_dim="time",  # dimension en la que concatenar
            # dim_order=["time", "lat", "lon"],
            # hay que ser explícitos y añadir data_vars para evitar un warning
            # data_vars='all' vs 'minimal':
            #  - 'all': Intenta concatenar todas las variables que encuentre en los archivos, aunque no
            #    tengan la dimensión de tiempo (esto es más lento y gasta más memoria).
            #  - 'minimal': Solo concatena las variables que realmente cambian entre archivos, es decir
            #    las que tienen la dimension a concatenar. Las variables que son iguales en todos (como
            #    la latitud/longitud) las deja tal cual, lo cual es mucho más eficiente.
            data_vars="minimal",
            # de nuevo, para evitar un warning, hay que ser explícitos con compat:
            #  - compat='no_conflicts': Es el comportamiento actual. Xarray comprueba que los valores de
            #    las variables que no se concatenan (como la latitud o la longitud) sean iguales en todos
            #    los archivos. Si hay una diferencia, te avisará con un error.
            #  - compat='override': Es el estándar del futuro. Xarray asume que las variables con el mismo
            #    nombre son idénticas en todos los archivos y usa las del primero que lee, sin perder tiempo
            #    comparándolas con los demás. Es ideal si sabes que tus mallas (grids) son exactamente iguales.
            compat="override",
            # y como he dicho que compat='override', no puedo dejar coords='different', que es la opción por
            # defecto en xarray (si las coordenadas son distintas entre archivos, concaténalas). Esto implica
            # que tiene que comparar las coordenadas, pero yo le he dicho con compat='override' que no pierda
            # tiempo en eso, luego hay una contradicción, que es de lo que se queja xarray si no ponemos
            # coords='minimal'.
            coords="minimal",
            parallel=True,  # habilita la lectura paralela (recomendada para muchos archivos grandes)
        )
