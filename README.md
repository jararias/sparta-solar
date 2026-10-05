
<p align="center">
<img src="https://raw.githubusercontent.com/jararias/sparta-solar/main/docs/images/sunny-helmet-black-transparent-recortada.png" alt="logo" width="50%">
</p>

# sparta-solar: Clear-sky Solar Irradiance with the SPARTA Radiative Transfer Model

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![Tests](https://raw.githubusercontent.com/jararias/sparta-solar/main/docs/images/tests-badge.svg)
![Coverage](https://raw.githubusercontent.com/jararias/sparta-solar/main/docs/images/coverage-badge.svg)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22164936.svg)](https://doi.org/10.5281/zenodo.22164936)

A Python library to compute clear-sky solar irradiance at the ground surface using the _Solar PArameterization of the Radiative Transfer of the Atmosphere_ ([SPARTA](http://hdl.handle.net/10630/28011)) radiative transfer model, with built-in access to multiple atmospheric databases (Copernicus [CAMS Radiative Service](https://confluence.ecmwf.int/display/CKB/CAMS+solar+radiation+time-series%3A+data+documentation) via [SODA](https://www.soda-pro.com/web-services/radiation/cams-radiation-service/info), hourly NASA [MERRA-2](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/) data via [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/NASA_GSFC_MERRA_aer_2?hl=es-419), and a daily dataset curated and maintained as part of the sparta-solar library on a dedicated Hugging Face [dataset](https://huggingface.co/datasets/josearuizarias/merra2-daily-clearsky)).

---

> **Licence:** sparta-solar is distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). It is free to use, share and adapt for research and other **non-commercial** purposes, with attribution and under the same licence. **Commercial use is not permitted** without a separate agreement with the author.

## Quick install

```bash
pip install sparta-solar
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add sparta-solar
```

Note that the distribution is named `sparta-solar`, but the Python package is imported as `spartasolar`:

```python
import spartasolar
```

## Quick examples

At individual sites:

```python
import pandas as pd
from spartasolar.atmosphere import merra2_daily

times  = pd.date_range("2020-06-15", periods=24, freq="h")
atmos  = merra2_daily.at_sites(
    times=times,
    latitude=36.72,  # or a sequence of latitudes...
    longitude=-4.42)  # ... and longitudes
result = atmos.compute()
print(result)
```

On a regular spatial grid:

```python
import numpy as np

lats = np.arange(-60, 60.1, 1)
lons = np.arange(-90, 90.1, 1)
atmos  = merra2_daily.on_regular_grid(
    times=times,
    latitude=lats,
    longitude=lons)
result = atmos.compute()
print(result)
```


## Documentation

Full documentation — installation guide, user guide, quick reference, and API reference — is available at **<https://jararias.github.io/sparta-solar>**

## Citation

```bibtex
@article{ruizarias2023sparta,
  author  = {Ruiz-Arias, Jose A.},
  title   = {{SPARTA}: Solar parameterization for the radiative transfer of the cloudless atmosphere},
  journal = {Renewable and Sustainable Energy Reviews},
  volume  = {188},
  pages   = {113833},
  year    = {2023},
  doi     = {10.1016/j.rser.2023.113833}
}
```

## License

[CC BY-NC-SA 4.0](LICENSE) — free for research and other non-commercial use, with attribution and under the same licence. Commercial use is not permitted without a separate agreement with the author.
