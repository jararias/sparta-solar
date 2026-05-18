# Installation

## Requirements

- Python 3.12 or newer
- An internet connection for the first download of MERRA-2 data files
  (the files are cached locally afterwards)
- Optional: a Google Earth Engine account (for the `merra2_gee` source)
- Optional: a SODA API registration (for the `crs_soda` source)

---

## Install from GitHub

=== "uv (recommended)"

    ```bash
    uv add git+https://github.com/jararias/sparta-solar
    ```

=== "pip"

    ```bash
    pip install git+https://github.com/jararias/sparta-solar
    ```

---

## Development installation

Clone the repository and install in editable mode with all optional dependencies:

```bash
git clone https://github.com/jararias/sparta-solar.git
cd sparta-solar
uv sync --all-extras
```

or with pip:

```bash
git clone https://github.com/jararias/sparta-solar.git
cd sparta-solar
pip install -e ".[dev]"
```

---

## Optional dependencies

| Extra | What it enables |
|---|---|
| `dev` | pytest, coverage, linters |
| `docs` | MkDocs, mkdocstrings |

Install an extra with:

```bash
pip install "sparta-solar[dev] @ git+https://github.com/jararias/sparta-solar"
```

---

## Verify the installation

```python
import spartasolar
print(spartasolar.__version__)
```

---

## Next steps

Now that sparta-solar is installed, read the [User Guide](user-guide.md) to
learn how to load atmospheric data and compute clear-sky irradiance.
