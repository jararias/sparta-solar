
from pathlib import Path

import pandas as pd
import pylab as pl

import spartasolar as sparta


def load_capitals():
    if not (path := Path("capitals_of_the_world.csv")).exists():
        url = ("https://gist.githubusercontent.com/ofou/df09a6834a8421b4f376c875194915c9/raw/"
               "355eb56e164ddc3cd1a9467c524422cb674e71a9/country-capital-lat-long-population.csv")
        pd.read_csv(url, sep=",").to_csv(path)
    return pd.read_csv(path, sep=",")

data = load_capitals().sort_values(by="Population", ascending=False)
data = data.iloc[:10:2]

rad = sparta.sites(
    times_utc=pd.date_range("2015-01-01", "2017-12-31", freq="h"),
    latitude=data.Latitude,
    longitude=data.Longitude,
    atmos="merra2_daily",
    site_names=data["Capital City"]
)

rad.ghi.resample(time="D").mean().swap_dims({"site": "name"}).plot.line(x="time", hue="name")
pl.show()

