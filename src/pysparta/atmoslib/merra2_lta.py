
import importlib
from pathlib import Path
from typing import Self, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from loguru import logger

from ..validation import Latitude, Longitude, validate_type
from ._base import BaseAtmosphere, make_cf_compliant


logger.disable(__name__)
logger = logger.opt(colors=True)


def get_database_path():
    return Path(importlib.resources.files("pysparta.atmoslib")) / "merra2_lta_data"


class MERRA2LTAAtmosphere(
    BaseAtmosphere,
    database_path=get_database_path()
):

    @classmethod
    def at_sites(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
        site_names: Sequence[str] | None = None,
    ) -> Self:

        latitude = np.asarray([validate_type(lat, Latitude) for lat in latitude], dtype=float).reshape(-1)
        longitude = np.asarray([validate_type(lon, Longitude) for lon in longitude], dtype=float).reshape(-1)

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

        def assign_year(ds, year):
            times_month_start = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
            return ds.assign_coords(month=times_month_start + pd.Timedelta(14.5, "d")).rename({"month": "time"})

        years = cls._infer_years_from_times(times)
        dataset_lta = xr.open_dataset(cls.database_path / "merra2_lta_data_1999-2018.nc", chunks={})
        dataset = xr.concat([assign_year(dataset_lta, year) for year in years], dim="time", data_vars="minimal")

        if "time" in dataset.coords:
            return dataset.chunk(time=12)  # monthly chunks
        return dataset
