---
language:
- en
license: cc-by-4.0
tags:
- climate
- meteorology
- merra2
- clearsky
- solar-radiation
- zarr
- atmospheric-science
pretty_name: MERRA-2 Daily Clearsky Dataset
size_categories:
- 10K<n<100K
---

# MERRA-2 Daily Clearsky Dataset

## Dataset Description
This dataset contains daily clearsky-related atmospheric and surface parameters derived from NASA's Modern-Era Retrospective analysis for Research and Applications, Version 2 (MERRA-2). The data covers the period from **1999 to 2018** globally, at its native resolution, and is stored in a cloud-optimized format to support the worldwide evaluation of clear-sky solar irradiance with the SPARTA model ([Ruiz-Arias, 2024](http://hdl.handle.net/10630/28011)) via **sparta-solar** for solar energy modeling, atmospheric research, and climatological studies.

- **Repository maintained by:** @josearuizarias
- **Primary Source:** NASA Global Modeling and Assimilation Office (GMAO)
- **Temporal Range:** 1999 - 2018
- **Temporal Resolution:** Daily
- **Spatial Coverage:** Global
- **Spatial Resolution:** 0.5° x 0.625° (Native MERRA-2 grid: 361 latitudes x 576 longitudes)

## Dataset Structure

### Data Fields
The dataset includes the following variables mapping key atmospheric and surface properties under clear-sky conditions:

* `albedo`: Surface albedo (dimensionless).
* `ozone`: Total-column ozone content ($kg \cdot m^{-2}$).
* `pwater`: Total-column water vapor content (precipitable water) ($kg \cdot m^{-2}$).
* `pressure`: Surface pressure ($Pa$).
* `altitude`: Surface height above sea level ($m$).
* `beta`: Angstrom turbidity coefficient (dimensionless).
* `alpha`: Angstrom wavelength exponent (dimensionless).
* `ssa`: Aerosol single-scattering albedo (dimensionless).
* `time`: Daily timestamp dimension.
* `lat`: Latitude coordinates (-90 to 90).
* `lon`: Longitude coordinates (-180 to 175.625).

## Data Origin and Methodology
The data is extracted from the official MERRA-2 products (e.g., [earthaccess](https://earthaccess.readthedocs.io/en/stable/)) and aggregated to daily time steps maintaining the native geographical grid of MERRA-2.

`albedo` is the *ALBEDO* MERRA-2 variable from collection *tavg1_2d_rad_Nx* (M2T1NXRAD) for radiation diagnostics, `pressure`, `ozone` and `pwater` are the variables *PS*, *TO3* and *TQV*, respectively, from collection *tavg1_2d_slv_Nx* (M2T1NXSLV) for single‐level diagnostics, and `beta`, `alpha` and `ssa` are evaluated from the variables *TOTEXTTAU*, *TOTSCATAU* and *TOTANGSTR* of collection *tavg1_2d_aer_Nx* (M2T1NXAER) for aerosol-related diagnostics. `alpha` is *TOTANGSTR*, `beta` is calculated from *TOTEXTTAU* and *TOTANGSTR* using the Angstrom's formula and `ssa` is the ratio of *TOTSCATAU* to *TOTEXTTAU*. 

## File Format and Usage
The dataset is structured into individual **Zarr stores per year** (e.g., `1999/`, `2000/`, etc.) that are not chunked. The reason is that they are conceived to be accessed from local using **sparta-solar** in yearly blocks, which is fast for both point and grid estimates of clear-sky solar irradiance.

## Citations & Acknowledgments

### How to cite this dataset
If you use this dataset in your academic or professional work, please cite both this repository and the core MERRA-2 reference:

```bibtex
@misc{merra2_daily_clearsky,
  author = {Ruiz-Arias, Jose A.},
  title = {MERRA-2 Daily Clearsky Dataset},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Data Repository},
  howpublished = {\url{https://huggingface.co}}
}

@article{merra2_official,
  author = {Gelaro, Ronald and others},
  title = {The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2)},
  journal = {Journal of Climate},
  volume = {30},
  number = {14},
  pages = {5419-5454},
  year = {2017},
  doi = {10.1175/JCLI-D-16-0758.1}
}
```

### License
This dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Use of the underlying raw data must adhere to NASA's Earth Science Data and Information Policy.
