
import functools
import inspect
import json
import sys
from math import ceil
from pathlib import Path

import blosc2
import numpy as np
import platformdirs
import xarray as xr
from loguru import logger
from zarr.codecs import BloscCodec, BitRound  #, BloscShuffle

from spartasolar import set_logger
from spartasolar.atmoslib._base import make_cf_compliant
from spartasolar.atmoslib.conversions import pwater_in_cm_to_kg_m2, ozone_in_cm_to_kg_m2

set_logger("INFO")


class TruncFilter:
    def __init__(self, precision, dtype='f4', astype='i4'):
        self._fill_value = -999
        self._data_type = dtype
        self._store_type = astype
        self._precision = int(precision)
        self._scale_factor = 10**self._precision

    def encode(self, data):
        values = self._scale_factor * np.round(data.astype(self._data_type), self._precision)
        return np.where(np.isnan(values), self._fill_value, values).astype(self._store_type)

    def decode(self, data):
        return (np.where(data == self._fill_value, np.nan, data) / self._scale_factor).astype(self._data_type)

@functools.lru_cache()
def load_metadata(path: Path) -> dict:
    with path.joinpath("metadata.json").open("r") as f:
        return json.load(f)

def load_array(path: Path, variable: str) -> np.ndarray:
    """do not remove. It is used by save_array to make consistency checks"""
    file_name = Path(path) / f'{variable}.bl2'
    metadata = json.load(open(Path(path) / 'metadata.json', 'r'))
    data = blosc2.load_array(file_name.as_posix())

    visible_modules = sys.modules[__name__]
    visible_class_names, visible_classes = zip(*inspect.getmembers(visible_modules, inspect.isclass))
    for filter_descr in metadata[variable]['filters']:
        filter_cls_name = filter_descr['class']
        filter_cls_kwargs = filter_descr['kwargs']
        if filter_cls_name not in visible_class_names:
            raise ValueError(f'unknown filter of type `{filter_cls_name}`')
        filter_cls = visible_classes[visible_class_names.index(filter_cls_name)]
        cfilter = filter_cls(**filter_cls_kwargs)
        data = cfilter.decode(data)

    return data

def export_blosc2_to_zarr_daily(blosc2_path: Path, zarr_path: Path, year: int) -> xr.Dataset:

    inpath = blosc2_path / f"{year}"
    outpath = zarr_path / f"{year}"
    logger.info(f"{inpath} -> {outpath}")

    # read the bl2 dataset...
    metadata = load_metadata(inpath)
    variables = [p.stem for p in inpath.glob("*.bl2")]

    # assert the (time, lat, lon) grid on all variables is identical, and retrieve it
    def get_start_stop_end(coord_name):
        def iter_coords():
            for variable in variables:
                if (bounds_dict := metadata[variable].get(coord_name)) is None:
                    continue
                if coord_name == "times":
                    bounds_dict["delta"] = tuple(bounds_dict["delta"])
                yield tuple(bounds_dict.values())
        return set(tuple([bounds for bounds in iter_coords()]))

    assert len(get_start_stop_end("times")) == 1, "all variables must have the same time bounds"
    assert len(get_start_stop_end("latitude")) == 1, "all variables must have the same latitude bounds"
    assert len(get_start_stop_end("longitude")) == 1, "all variables must have the same longitude bounds"

    # time coordinates...
    start, stop, delta = get_start_stop_end("times").pop()
    one_ns = np.timedelta64(1, 'ns')
    times_utc = np.arange(start, np.datetime64(stop) + one_ns, np.timedelta64(*delta))

    # latitude coordinates...
    start, stop, step = get_start_stop_end("latitude").pop()
    latitudes = np.arange(start, stop+1e-6, step)

    # longitude coordinates...
    start, stop, step = get_start_stop_end("longitude").pop()
    longitudes = np.arange(start, stop+1e-6, step)

    dims = {"elevation": ("lat", "lon")}  # else ("dim", "lat", "lon")
    coords = {"time": ("time", times_utc), "lat": ("lat", latitudes), "lon": ("lon", longitudes)}
    data_vars = {var: (dims.get(var, ("time", "lat", "lon")), load_array(inpath, var)) for var in variables}

    ds = xr.Dataset(data_vars, coords)

    # The blosc2 dataset has a 3-day buffer at the beginning and end of the year to prevent
    # time extrapolations. The problem is that for multi-anual datasets the bounds between
    # consecutive years overlap and I need to manage this manually (triming the overlapping
    # days). With the zarr dataset and a proper time chunking, I can skip this strategy and
    # simply load one more year before and after the requested years, if I can anticipate
    # extrapolation issues, and let Dask load only the chunks that are requested.
    ds = ds.sel(time=str(year))
    logger.success(f"xarray.Dataset loaded with sizes: {ds.sizes}")

    ds = ds.rename({"elevation": "altitude"})
    ds["ozone"] = ozone_in_cm_to_kg_m2(ds["ozone"])
    ds["pwater"] = pwater_in_cm_to_kg_m2(ds["pwater"])
    ds["pressure"] = ds["pressure"] * 100  # hPa -> Pa
    logger.success("units conversions applied")

    ds = make_cf_compliant(ds)
    ds.attrs["references"] = "doi:10.5067/KLICLTZ8EM9D, doi:10.5067/Q9QMY5PBNV1T, doi:10.5067/VJAFPLI1CSIV"

    logger.success("dataset made CF compliant")

    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True, exist_ok=True)

    def get_filters(variable):
        # keepbits es el número de dígitos de la mantisa a conservar cuando usamos BitRound.
        # Ojo, no es exactamente la precisión (es decir, el número de decimales). Si queremos
        # mantener D decimales: keepbits = ceil(log2(10^D)) = ceil(D * log2(10)) ~= ceil(D * 3.32)
        # Así: 4 bits ~= 1 decimal, 7 bits ~= 2 decimals, 10 bits ~= 3 decimals
        decimals = {"albedo": 3, "pressure": 1, "ozone": 3, "pwater": 3,
                    "alpha": 2, "beta": 3, "ssa": 3, "altitude": 1}
        if variable in decimals:
            return [BitRound(keepbits=ceil(3.32*decimals.get(variable)))]
        return None

    # chunk_mapping = {"time": ds.sizes["time"], "lat": 30, "lon": 30}  # optimizado para extraer series puntuales
    # chunk_mapping = {"time": ds.sizes["time"], "lat": 180, "lon": 288}  # optimizado para extraer series puntuales
    # los dos anteriores no funcionan... van súper lentos. Se ve que, trabajando en local, es más eficiente cargar
    # en memoria ficheros anuales de 300 MB que tener muchos ficheros más pequeños. Lo de la optimizción es un latazo
    chunk_mapping = {"time": ds.sizes["time"], "lat": ds.sizes["lat"], "lon": ds.sizes["lon"]}  # optimizado para extraer series puntuales
    ds_chunked = ds.chunk(chunk_mapping)

    compressor = BloscCodec(cname="zstd", clevel=9)
    encoding = {var: {"compressors": compressor,
                      "filters": get_filters(var),
                      "chunks": [chunk_mapping[dim] for dim in ds[var].dims]}
                for var in ds.data_vars}

    ds_chunked.to_zarr(outpath, encoding=encoding, zarr_format=3, mode="w")

    return ds_chunked

if __name__ == "__main__":

    year = int(sys.argv[1])  # 2016
    blosc2_path = platformdirs.user_data_path("sparta-solar/merra2-daily-blosc2")
    zarr_path = platformdirs.user_data_path("sparta-solar/merra2-daily")
    data = export_blosc2_to_zarr_daily(blosc2_path, zarr_path, year)
