
from typing import Self, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from .merra2_lta import MERRA2LTAAtmosphere, get_database_path as lta_database_path

logger.disable(__name__)


def get_database_path():
    return lta_database_path()


class MERRA2CDAAtmosphere(
    MERRA2LTAAtmosphere,
    database_path=get_database_path()):

    @classmethod
    def at_sites(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
        site_names: Sequence[str] | None = None,
    ) -> Self:

        dataset = super().sites(
            times=times,
            latitude=latitude,
            longitude=longitude,
            site_names=site_names)

        if "pwater" in dataset:
            dataset["pwater"] = 0.1

        if "beta" in dataset:
            dataset["beta"] = 0.01

        return dataset

    @classmethod
    def on_regular_grid(
        cls,
        times: np.ndarray[tuple[int], np.datetime64] | pd.DatetimeIndex,
        latitude: Sequence[float],
        longitude: Sequence[float],
    ) -> Self:

        dataset = super().on_regular_grid(
            times=times,
            latitude=latitude,
            longitude=longitude)

        if "pwater" in dataset:
            dataset["pwater"] = 0.1

        if "beta" in dataset:
            dataset["beta"] = 0.01

        return dataset
