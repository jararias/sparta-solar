# About the Models 📚

sparta-solar includes two primary broadband clear-sky models. While both provide high-quality solar irradiance estimates, they differ in their physical complexity and spectral resolution.

## SPARTA Model
**Solar Parameterization of the Radiative Transfer of the Atmosphere**

SPARTA is the flagship model of this library. It is a **two-band** model that separates the solar spectrum into:
- **UV-VIS**: 280 nm – 700 nm
- **Near-IR**: 700 nm – 4000 nm

### Why use SPARTA?
- **Better Accuracy**: By using two spectral bands, it handles the different absorption characteristics of water vapor and aerosols much more precisely than single-band models.
- **Modern Physics**: It uses updated parameterizations for gas transmittances and aerosol scattering.
- **Circumsolar Support**: Native support for circumsolar irradiance (CSI) corrections, essential for concentrated solar power (CSP) applications.

!!! success "Recommended"
    SPARTA is the recommended model for most high-precision scientific and engineering applications.

---

## BIRD Model
**The Classic Standard (Bird & Hulstrom, 1981)**

The Bird model is a widely recognized **single-band** broadband model. It has been a benchmark in the solar community for decades.

### Why use BIRD?
- **Benchmark Consistency**: Ideal for comparing results with historical studies or legacy software.
- **Simplicity**: Lower computational overhead (though negligible in modern systems).
- **Validation**: Extensively validated against thousands of ground stations worldwide.

---

## Technical Comparison


| Feature | SPARTA | BIRD |
| :--- | :--- | :--- |
| **Bands** | 2 (UV-VIS, NIR) | 1 (Broadband) |
| **Solar Constant** | 1361.1 W/m² | 1353 W/m² |
| **Circumsolar** | Native Support | Not included |
| **Aerosols** | Advanced \(\alpha, \beta, ssa, asy\) | Simplified \(\tau_a\) |

## How to Cite

If you use these models in your research, please cite:

**For SPARTA:**
> Arias, J. R., & Ruiz-Arias, J. A. (2025). Solar Parameterization of the Radiative Transfer of the Atmosphere (SPARTA): A two-band broadband clear-sky solar radiation model. *Solar Energy*. [DOI: 10.1016/j.solener.2024.112836](https://doi.org)

**For BIRD:**
> Bird, R. E., & Hulstrom, R. L. (1981). A simplified clear sky model for direct and diffuse insolation on horizontal surfaces. *SERI Technical Report*.
