r"""Bird & Hulstrom Clear-Sky Broadband Model.

This module implements the Bird & Hulstrom (1981) model, a widely used 
broadband parameterization for calculating clear-sky solar irradiance. 
Unlike SPARTA, Bird is a single-band model that estimates direct and 
diffuse components based on a set of individual transmittance functions.
"""
import numpy as np

from .sandbox import cast_to_compatible_arrays as cast_arrays


def BIRD(
    cosz: float | np.ndarray = 0.5,
    pressure: float | np.ndarray = 1013.25,
    albedo: float | np.ndarray = 0.2,
    pwater: float | np.ndarray = 1.4,
    ozone: float | np.ndarray = 0.3,
    beta: float | np.ndarray = 0.1,
    alpha: float | np.ndarray = 1.3,
    ssa: float | np.ndarray = 0.92,
    asy: float | np.ndarray = 0.65,
    ecf: float | np.ndarray = 1.0
) -> dict[str, np.ndarray]:
    r"""Calculates solar irradiance using the Bird & Hulstrom model.

    The model estimates solar radiation through individual transmittance 
    processes for Rayleigh scattering, ozone absorption, uniformly mixed gases, 
    water vapor, and aerosol extinction. It also accounts for multiple 
    reflections between the ground and the sky.

    Args:
        cosz: Cosine of the solar zenith angle.
        pressure: Atmospheric surface pressure in hPa.
        albedo: Ground surface albedo (0 to 1).
        pwater: Precipitable water in cm.
        ozone: Ozone vertical pathlength in atm-cm (1 atm-cm = 1000 DU).
        beta: Ångström's turbidity coefficient (AOD at 1000 nm).
        alpha: Ångström's wavelength exponent.
        ssa: Aerosol single-scattering albedo at ~700 nm.
        asy: Aerosol asymmetry parameter.
        ecf: Eccentricity correction factor for the Sun-Earth orbit.

    Returns:
        dict[str, np.ndarray]: A dictionary containing:
            - `dni`: Direct normal irradiance [W/m²].
            - `dhi`: Direct horizontal irradiance [W/m²].
            - `dif`: Diffuse horizontal irradiance [W/m²].
            - `ghi`: Global horizontal irradiance [W/m²].

    Notes:
        - The model uses a fixed solar constant (\(G_{sc}\)) of 1353 W/m².
        - Nighttime values are automatically masked (set to 0) for zenith 
          angles greater than 90.5°.
        - The algorithm includes a 0.9662 correction factor for the 
          direct normal component as per the original publication.

    References:
        - Bird, R. E., & Hulstrom, R. L. (1981). A simplified clear sky 
          model for direct and diffuse insolation on horizontal surfaces. 
          Solar Energy Research Institute (SERI).
    """

    cosz, pressure, albedo, pwater, ozone, beta, alpha, ssa, asy, ecf, restore_shape = \
        cast_arrays(cosz, pressure, albedo, pwater, ozone, beta, alpha, ssa, asy, ecf)

    INP_SHAPE = cosz.shape
    COSZ_MIN = np.cos(np.radians(90.5))
    SC = 1353.  # W/m2, solar constant

    nighttime = cosz <= COSZ_MIN

    def notna(ar):
        return (~np.isnan(ar)) & (ar != -999) & (~nighttime)

    domain = (
        notna(cosz) & notna(ecf) & notna(pressure) &
        notna(ozone) & notna(pwater) & notna(albedo) &
        notna(beta) & notna(alpha) & notna(ssa) & notna(asy)
    )

    # .. initialize outputs
    Ebn = np.full(INP_SHAPE, np.nan)  # direct normal irradiance, W/m2
    Ebh = np.full(INP_SHAPE, np.nan)  # direct horizontal irradiance, W/m2
    Edh = np.full(INP_SHAPE, np.nan)  # diffuse horizontal irradiance, W/m2
    Egh = np.full(INP_SHAPE, np.nan)  # global horizontal irradiance, W/m2

    # .. airmass
    c = [0.48353, 0.095846,  96.741, 1.7540]
    sza = np.degrees(np.arccos(cosz))[domain]
    am = np.maximum(1., 1. / (cosz[domain] + c[0]*(sza**c[1])/((c[2]-sza)**c[3])))
    amr = am * (pressure[domain] / 1013.25)

    # DIRECT IRRADIANCE...

    TR = np.clip(np.exp(-0.0903*(amr**.84)*(1.+amr-amr**1.01)), 0., 1.)
    uo = am*ozone[domain]
    To = np.clip(
        1 - (0.1611*uo/((1.+139.48*uo)**0.3035) -
             0.002715*uo/((1.+(0.044*uo))+0.0003*uo**2)), 0., 1.)
    Tg = np.clip(np.exp(-0.0127*amr**0.26), 0., 1.)
    uw = am*pwater[domain]
    Tw = np.clip(1 - 2.4959*uw / (((1 + 79.034*uw)**0.6828) + 6.385*uw), 0., 1.)
    taua = beta[domain]*(0.2758*(0.38**(-alpha[domain])) + 0.35*(0.5**(-alpha[domain])))
    Ta = np.clip(np.exp(-(taua**0.873)*(1+taua-taua**0.7088)*am**0.9108), 0., 1.)

    Ebn[domain] = np.clip(0.9662*SC*ecf[domain]*TR*To*Tg*Tw*Ta, 0., np.inf)
    Ebh = Ebn*cosz

    # DIFFUSE IRRADIANCE...

    Taa = np.clip(1-(1-ssa[domain])*(1-am+am**1.06)*(1-Ta), 0., 1.)
    Tas = Ta/Taa
    Tabs = To*Tg*Taa*Tw
    Ba = 0.5*(1+asy[domain])
    rhos = 0.0685 + (1-Ba)*(1-Tas)
    Ed0h = SC*cosz[domain]*np.clip(0.79*Tabs*(0.5*(1-TR) + Ba*(1-Tas))/(1.-am+am**1.02), 0., 1.)

    Egh[domain] = (Ebh[domain] + Ed0h)/(1-albedo[domain]*rhos)
    Edh[domain] = Egh[domain] - Ebh[domain]

    # .. mask nighttime
    Ebn[nighttime] = 0.
    Ebh[nighttime] = 0.
    Edh[nighttime] = 0.
    Egh[nighttime] = 0.

    Ebn = restore_shape(Ebn)
    Ebh = restore_shape(Ebh)
    Edh = restore_shape(Edh)
    Egh = restore_shape(Egh)

    return {"dni": Ebn, "dhi": Ebh, "dif": Edh, "ghi": Egh}
