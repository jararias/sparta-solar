
import copy
from typing import Any, Sequence

import numpy as np
import pandas as pd


def ensure_tz_aware_datetime_index(times: Any, utc: bool = False) -> pd.DatetimeIndex:
    times_dti = copy.deepcopy(times)

    # 1. convert to DatetimeIndex, if not already
    if not isinstance(times_dti, pd.DatetimeIndex):
        if isinstance(times, str) or not isinstance(times, Sequence):
            times_dti = [times_dti]
        times_dti = pd.to_datetime(times_dti)

    # 2. ensure tz-aware, if not already
    if times_dti.tz is None:
        times_dti = times_dti.tz_localize("UTC")

    # 3. convert to UTC, if requested
    if utc:
        return times_dti.tz_convert("UTC")
    return times_dti


def pwater_in_kg_m2_to_cm(kg_m2: np.ndarray[float]) -> np.ndarray[float]:
    """Convert precipitable water in kg/m² to cm.

    Args:
        kg_m2 (np.ndarray[float]): The values of precipitable water in kg/m².

    Returns:
        np.ndarray[float]: The values in cm.
    """
    mm_layer = kg_m2
    return mm_layer * 1e-1

def pwater_in_cm_to_kg_m2(cm: np.ndarray[float]) -> np.ndarray[float]:
    """Convert precipitable water in cm to kg/m².

    Args:
        cm (np.ndarray[float]): The values of precipitable water in cm.

    Returns:
        np.ndarray[float]: The values in kg/m².
    """
    kg_m2 = cm * 1e1
    return kg_m2

def ozone_in_du_to_kg_m2(du: np.ndarray[float]) -> np.ndarray[float]:
    """Convert total column ozone amount in Dobson Units (DU) to kg/m².

    Args:
        du (np.ndarray[float]): The values of total column ozone in DU.
    
    Note:
        1 DU corresponds to a layer of gas that would be 0.01 mm thick at standard temperature
        and pressure (STP). The density of ozone at STP is approximately 2.1415 kg/m³. Hence:
        1 DU = 0.01 mm * 2.1415 kg/m³ = 2.1415e-5 kg/m²

    Returns:
        np.ndarray[float]: The values in kg/m².
    """
    return du * 2.1415e-5

def ozone_in_cm_to_kg_m2(cm: np.ndarray[float]) -> np.ndarray[float]:
    """Convert total column ozone amount in cm to kg/m².

    Args:
        cm (np.ndarray[float]): The values of total column ozone in cm.
    
    Note:
        1 DU corresponds to a layer of gas that would be 0.01 mm thick at standard temperature
        and pressure (STP). The density of ozone at STP is approximately 2.1415 kg/m³. Hence:
        1 DU = 0.01 mm * 2.1415 kg/m³ = 2.1415e-5 kg/m²
        1 cm = 1000 * 2.1415e-5 kg/m² = 2.1415e-2 kg/m² 

    Returns:
        np.ndarray[float]: The values in kg/m².
    """
    return cm * 2.1415e-2

def ozone_in_kg_m2_to_cm(kg_m2: np.ndarray[float]) -> np.ndarray[float]:
    """Convert total column ozone amount in kg/m² to cm.

    Args:
        kg_m2 (np.ndarray[float]): The values of total column ozone in kg/m².
    
    Note:
        For ozone, 1 DU is approximately equal to 2.1415e-5 kg/m². 

    Note:
        1 DU corresponds to a layer of gas that would be 0.01 mm thick at standard temperature
        and pressure (STP). The density of ozone at STP is approximately 2.1415 kg/m³. Hence:
        1 DU = 0.01 mm * 2.1415 kg/m³ = 2.1415e-5 kg/m²
        1 cm = 1000 * 2.1415e-5 kg/m² = 2.1415e-2 kg/m² 

    Returns:
        np.ndarray[float]: The values in cm.
    """
    return kg_m2 / 2.1415e-2