This directory contains small parquet data files used by the test suite.

Data is derived from the default user data cache (~/.local/share/sparta-solar/)
and covers a single site (lat=37.5°N, lon=3.5°W) for the year 2015.

## Files

### merra2_gee/
- `merra2_gee_hourly_375000N_35000W_2015.parquet`
  MERRA-2 hourly atmospheric data for 2015 retrieved via Google Earth Engine.
  Columns: times_utc, albedo, pressure, ozone, pwater, beta, alpha, ssa

### crs_soda/
- `crs_soda_mcclear_v1.0.0_375000N_35000W_2015.parquet`
  CRS SODA McClear atmospheric data for 2015.
  Columns: times_utc, pressure, albedo, pwater, ozone, aod550, beta, alpha

## Usage in tests

Tests point to this directory using the `merra2_gee.data_dir` and
`crs_soda.data_dir` config options so no network access is required.
