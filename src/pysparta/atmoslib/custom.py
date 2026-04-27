
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
    
        obj = cls()
        obj._atmosphere = build_atmosphere_on_regular_grid(
            times=times,
            latitude=latitude,
            longitude=longitude,
            constituents=constituents,
            var_attrs=var_attrs,
            global_attrs=global_attrs)
        return obj

