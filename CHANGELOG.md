## v0.2.3
#### October, 2026
- Bug fixes:
  - `compute()` failed for atmospheres with more than one site (`validate_site_names` compared a numpy array with a list)
  - `compute(include_atmosphere=True)` is now implemented: the atmospheric constituents are merged into the output
  - `custom` atmosphere: `pwater` (cm) and `ozone` (atm-cm) are now converted to kg m-2 when stored. Before, they were interpreted as kg m-2, so the results were wrong
  - `merra2_cda`: precipitable water was set to 0.01 cm instead of 0.1 cm, and the variable attributes were lost
  - `database_path` is now resolved lazily, so `config.set_option("<source>.data_dir", ...)` after import is honoured
  - `merra2_daily`: no padding years outside 1999-2018 near the period boundaries; `list_merra2_daily_cache` and `clear_merra2_daily_cache` now find the cached years; NaN albedo is filled in `at_sites` as in `on_regular_grid`
  - `merra2_gee` and `crs_soda`: no padding years outside the available period
  - `crs_soda`: robust handling of missing `alpha` column and missing altitude in the service response
  - CF global attribute `title` was written as `title:` in `crs_soda` and `merra2_gee`
  - Bird model: sign error in the ozone transmittance; NaN at very large air masses
  - SPARTA and Bird: no negative irradiance between the horizon and the night-time threshold (cosz > cos(90.5°))
  - SPARTA: invalid values of `csi_param` raise an error (they silently disabled the circumsolar irradiance); it is now case-insensitive
  - circular import in `spartasolar.validation` and broken `Atmosphere` type alias removed
- Documentation:
  - fixed examples (`site_name` in `crs_soda.at_site`, missing `import numpy as np`, Bird example in `compute`)
  - consistent coverage and periods (`merra2_daily` 1999-2018, CAMS McClear global)
  - documented `merra2_cda` (clean and dry atmosphere), `include_atmosphere` and the array shapes and units of `custom`
  - clarified that the distribution is `sparta-solar` but the package is imported as `spartasolar`
  - licence terms (CC BY-NC-SA 4.0, non-commercial use) stated in the README and docs home page
- Tests: new tests for multi-site `compute()`, `include_atmosphere`, `custom` units, `merra2_cda` and model edge cases; tests no longer depend on the order in which they run or on the user's local cache

## v0.2.2
#### August, 2026
- Added `CITATION.cff` and Zenodo DOI (10.5281/zenodo.22164936)


## v0.2.1
#### July, 2026
- Fix deprecation warnings and vulnerabilities raised by third-party packages:
  - change "d" by "D" in pands Timedelta
  - tornado>=6.5.7 (tornado==6.5.5 is introduced via panel, geoviews and hvplot)
  - msgpack>=1.2.1 (msgpack==1.1.2 is introduced via blosc==4.2.0)
  - cryptography>=48.0.1 (cryptography==48.0.0 is introduced via earthengine-api==1.7.26)
  - bleach>=6.4.0 (bleach==6.3.0 is introduced via panel, geoviews and hvplot)

## v0.2.0
#### June, 2026
- General code review to fix minor bugs
- Improved timezone handling in time objects

## v0.1.0
#### May, 2025
- First release
