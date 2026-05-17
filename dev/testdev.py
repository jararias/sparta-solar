
import numpy as np
import pandas as pd
import pylab as pl  # noqa: F401

from spartasolar import SPARTA
from spartasolar.benchmark import BIRD  # noqa: F401

# TODO Inline documentation to explain the input arguments

times = pd.date_range('2024-01-01T10', '2024-01-01T16', freq='1h')
lats = np.linspace(30, 50, 15)
lons = np.linspace(-12, 8, 15)

res = SPARTA(times, sites={'latitude': lats, 'longitude': lons},
              atmos='merra2_cda')
