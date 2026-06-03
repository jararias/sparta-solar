# sparta-solar

**Solar PArameterization of the Radiative Transfer of the Atmosphere**

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Tests](docs/images/tests-badge.svg)](docs/images/tests-badges.svg)
![Tests](https://raw.githubusercontent.com/jararias/sparta-solar/main/docs/images/tests-badge.svg)
[![Coverage](docs/images/coverage-badge.svg)](docs/images/coverage-badge.svg)
![Coverage](https://raw.githubusercontent.com/jararias/sparta-solar/main/docs/images/coverage-badge.svg)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

A Python library for computing clear-sky solar irradiance using the SPARTA
radiative transfer model, with integrated access to multiple atmospheric databases
(NASA MERRA-2, Copernicus CRS/SODA, Google Earth Engine).

---

## Quick install

```bash
pip install sparta-solar
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add sparta-solar
```

## Quick example

```python
import pandas as pd
from spartasolar.atmosphere import merra2_daily

times  = pd.date_range("2020-06-15", periods=24, freq="h")
atm    = merra2_daily.at_sites(times=times, latitude=36.72, longitude=-4.42)
result = atm.compute(model="SPARTA")
print(result)
```

## Documentation

Full documentation — installation guide, user guide, quick reference, and API
reference — is available at:

**<https://jararias.github.io/sparta-solar>**

## Citation

```bibtex
@article{ruizarias2023sparta,
  author  = {Ruiz-Arias, Jose A.},
  title   = {{SPARTA}: Solar parameterization for the radiative transfer
             of the cloudless atmosphere},
  journal = {Renewable and Sustainable Energy Reviews},
  volume  = {188},
  pages   = {113833},
  year    = {2023},
  doi     = {10.1016/j.rser.2023.113833}
}
```

## License

[CC BY-NC-SA 4.0](LICENSE) — free for non-commercial use with attribution.
