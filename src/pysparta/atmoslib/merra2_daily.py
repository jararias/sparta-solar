
from pathlib import Path
from typing import Self, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import platformdirs
import xarray as xr
from loguru import logger

from ._base import BaseAtmosphere, make_cf_compliant
from ..config import get_option
from ..validation import Latitude, Longitude, validate_type

logger.disable(__name__)
logger = logger.opt(colors=True)


def get_database_path():
    user_path = (get_option("merra2_daily.data_dir") or
                 platformdirs.user_data_path('pysparta/merra2-daily'))
    if not user_path.exists():
        user_path.mkdir(parents=True, exist_ok=True)
    return user_path


class MERRA2DailyAtmosphere(
    BaseAtmosphere,
    database_path=get_database_path()
):

    @classmethod
    def at_sites(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float] | float,
        longitude: Sequence[float] | float,
        site_names: Sequence[str] | None = None,
    ) -> Self:

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

        if site_names is not None:
            output_dataset = output_dataset.assign_coords(
                site_name=("site", [site_names] if isinstance(site_names, str) else site_names))

        output_dataset = output_dataset.compute()

        obj = cls()
        obj._atmosphere = make_cf_compliant(output_dataset, overwrite=True)
        return obj

    @classmethod
    def on_regular_grid(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
    ) -> Self:

        latitude = np.asarray([validate_type(lat, Latitude) for lat in latitude], dtype=float).reshape(-1)
        longitude = np.asarray([validate_type(lon, Longitude) for lon in longitude], dtype=float).reshape(-1)

        # load the dataset. Check for local availability. If not available, download.
        dataset = cls._load_dataset(times)
        dataset["albedo"] = dataset["albedo"].fillna(0.)

        # lat-lon interpolation
        output_dataset = dataset.interp(lat=latitude, lon=longitude, method='linear')

        # time interpolation
        if "time" in output_dataset.coords:
            output_dataset = output_dataset.interp(time=times, method='quadratic')

        output_dataset = output_dataset.compute()

        obj = cls()
        obj._atmosphere = make_cf_compliant(output_dataset, overwrite=True)
        return obj

    @staticmethod
    def _infer_years_from_times(times: npt.NDArray[np.datetime64] | pd.DatetimeIndex) -> list[int]:
        the_times = times if isinstance(times, pd.DatetimeIndex) else pd.to_datetime(times)
        years = set(the_times.year)
        if (the_times[0] - pd.to_datetime(f"{min(years)}-01-01 12")) < pd.Timedelta(3, "D"):
            years.add(min(years)-1)
        if (pd.to_datetime(f"{max(years)}-12-31 12") - the_times[-1]) < pd.Timedelta(3, "D"):
            years.add(max(years)+1)
        return sorted(years)

    @staticmethod
    def _ensure_all_paths_are_local(paths: list[Path]) -> None:
        for path in paths:
            if not path.exists():
                logger.warning(f"missing path `{path}`. Fetching the dataset")
                raise NotImplementedError(f"missing path `{path}`. Fetching the dataset is not implemented yet")

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
