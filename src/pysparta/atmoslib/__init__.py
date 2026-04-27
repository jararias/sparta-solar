
import dataclasses

from .crs_sodaapi import CRSSODAAtmosphere
from .custom import CustomAtmosphere
from .merra2_geeapi import MERRA2GEEAtmosphere
from .merra2_cda import MERRA2CDAAtmosphere
from .merra2_lta import MERRA2LTAAtmosphere
from .merra2_daily import MERRA2DailyAtmosphere

custom_atmosphere = CustomAtmosphere()

@dataclasses.dataclass
class Atmosphere:
    crs_soda = CRSSODAAtmosphere()
    merra2_gee = MERRA2GEEAtmosphere()
    merra2_cda = MERRA2CDAAtmosphere()
    merra2_lta = MERRA2LTAAtmosphere()
    merra2_daily = MERRA2DailyAtmosphere()
