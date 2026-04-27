# Configuration Guide

Pysparta requires minimal setup, but you must provide your **SoDA User Email** to access atmospheric data from the CAMS Radiation Service.

## Configuration Methods

You can manage your settings in two ways: via a persistent configuration file or dynamically during execution.

### 1. Persistent Setup (Recommended)

Pysparta stores its settings in a `config.toml` file. To find out where this file should be located on your system, run:

```python
import pysparta
print(pysparta.get_config_path())
```

#### Default Location
- **Linux:** `~/.config/pysparta/config.toml`
- **macOS:** `~/Library/Application Support/pysparta/config.toml`
- **Windows:** `%APPDATA%\pysparta\config.toml`

#### Example `config.toml`
If the file doesn't exist, Pysparta will create a template for you. Edit it with your details:

```toml
# pysparta configuration file
soda_user_email = "your_registered_email@example.com"
data_dir = "~/pysparta_data"
```

---

### 2. Runtime Setup

If you prefer to set options programmatically (e.g., in a Jupyter Notebook or a script), use `set_option`:

```python
import pysparta

# Set your credentials for the current session
pysparta.set_option("soda_user_email", "user@example.com")

# Change the data storage directory
pysparta.set_option("data_dir", "/path/to/custom/storage")
```

!!! warning "Note"
    Changes made with `set_option` are **temporary** and will be lost when the Python session ends. For permanent changes, use the `config.toml` file.

## Available Options


| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `soda_user_email` | `str` | `"__not_set__"` | Your registered email at [soda-pro.com](https://soda-pro.com). |
| `data_dir` | `Path` | `"__not_set__"` | Directory where downloaded `.parquet` and `.bl2` files are cached. |

## Inspecting Current Config

You can check your active configuration at any time:

```python
import pysparta
pysparta.show_options()
```

---

## Technical Reference

The following functions manage the internal state of the configuration:

::: pysparta.config
    options:
      show_root_heading: false
      show_source: false

## Obtaining SoDA Credentials

To use Pysparta's data retrieval features, you need a free account from **SoDA-Pro**. This service provides the CAMS Radiation Service (CRS) data used for atmospheric characterization.

1.  **Register:** Go to the [SoDA-Pro Registration Page](https://soda-pro.com).
2.  **Activate:** Confirm your account via the email they send you.
3.  **Use your email:** The email address you used for registration is your `soda_user_email`.

!!! info "Free Service"
    The CAMS McClear service is generally free for registered users, but requires this email to track API usage and enforce quotas.

## Linear distances vs angular distances

Para distancias por debajo de unos 100 km se pueden establecer relaciones de conversión sencillas:

$\Delta x = 111.32 \; \cos(\phi) \; \Delta lon$, in km, where $\phi$ is the latitude

$\Delta y = 111.12 \; \Delta lat$, in km

$\Delta = \sqrt{(\Delta x)^2 + (\Delta y)^2}

Por ejemplo, considerando que la resolución de MERRA-2 es 0.5º x 0.625º, la longitud de la hipotenusa
de la celda varía con la latitud:

lat  hypot (km)
  0   89.0
 10   88.2
 20   85.8
 30   82.0
 40   77.0
 50   71.3
 60   65.6
 70   60.4
 80   56.9
 90   55.6

 Los datos puntuales de crs_soda y merra2_gee se recuperan y guardan truncando los decimales de latitude
 y longitude a cuatro cifras decimales. ¿Qué error "lineal" implica este trucamiento? Eso se puede calcular
 en función de la latitud y las cifras decimales (n) que se mantienen.

        Error (m)
lat   n=4  n=3   n=2
  0    16  157  1573
 10    16  156  1561
 20    15  152  1526
 30    15  147  1471
 40    14  140  1401
 50    13  132  1322
 60    12  124  1243
 70    12  117  1175
 80    11  112  1128
 90    11  111  1111

Desde el punto de vista de la resolución de los modelos globales actuales, y considerando únicamente la
resolución, considerar únicamente dos cifras decimales puede parecer suficiente pues el error de posicionamiento
es muy inferior a la resolución espacial típica de los modelos. No obstante, cuando se trata de extraer datos de
las rejillas elegir 2, 3 o 4 cifras decimales puede resultar en series diferentes, aunque parecidas por corresponder
a celdas adyacentes del modelo.