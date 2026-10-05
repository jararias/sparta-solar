# Installation

## Install from PyPI

=== "uv (recommended)"

    ```bash
    uv add sparta-solar
    ```

=== "pip"

    ```bash
    pip install sparta-solar
    ```


!!! note
    The distribution is named `sparta-solar`, but the Python package is imported as `spartasolar`:

    ```python
    import spartasolar
    print(spartasolar.__version__)
    ```

## Next steps

Now that sparta-solar is installed, read the [User Guide](user-guide.md) to
learn how to load atmospheric data and compute clear-sky irradiance.
