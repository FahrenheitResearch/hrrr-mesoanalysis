"""Surface weather parameter computations using metrust engine.

Heat Index, Wind Chill, Fosberg Fire Weather Index, Hot-Dry-Windy Index,
and Wet Bulb Globe Temperature (WBGT).

All functions take a ModelData dataclass and return dicts of 2D [ny, nx] arrays.
"""

import numpy as np
from metrust._metrust import calc as _calc


# Vectorized wrappers for scalar metrust functions
_heat_index_v = np.vectorize(_calc.heat_index, otypes=[np.float64])
_windchill_v = np.vectorize(_calc.windchill, otypes=[np.float64])
_fosberg_v = np.vectorize(_calc.fosberg_fire_weather_index, otypes=[np.float64])
_hot_dry_windy_v = np.vectorize(_calc.hot_dry_windy, otypes=[np.float64])


def compute_heat_index(data):
    """Compute heat index in Fahrenheit.

    Uses metrust heat_index(temperature_c, relative_humidity_pct).
    Only meaningful when T >= ~80F (27C) and RH >= ~40%.

    Returns dict with key: heat_index (2D [ny, nx] in F)
    """
    t2_C = (data.t2m_K - 273.15).astype(np.float64)
    td2_C = (data.td2m_K - 273.15).astype(np.float64)

    # Compute RH from T and Td using Magnus formula
    rh = 100.0 * np.exp(17.625 * td2_C / (243.04 + td2_C)) / \
         np.exp(17.625 * t2_C / (243.04 + t2_C))
    rh = np.clip(rh, 0.0, 100.0)

    # Heat index (returns Fahrenheit)
    hi = _heat_index_v(t2_C, rh)

    return {"heat_index": hi}


def compute_windchill(data):
    """Compute wind chill in Fahrenheit.

    Uses metrust windchill(temperature_c, wind_speed_ms).
    Only meaningful when T <= ~50F (10C) and wind > ~3 mph.

    Returns dict with key: windchill (2D [ny, nx] in F)
    """
    t2_C = (data.t2m_K - 273.15).astype(np.float64)
    wspd_ms = np.hypot(data.u10, data.v10).astype(np.float64)

    wc = _windchill_v(t2_C, wspd_ms)

    return {"windchill": wc}


def compute_fosberg_ffwi(data):
    """Compute Fosberg Fire Weather Index.

    Uses metrust fosberg_fire_weather_index(t_f, rh, wspd_mph).

    Returns dict with key: ffwi (2D [ny, nx], unitless)
    """
    t2_C = (data.t2m_K - 273.15).astype(np.float64)
    td2_C = (data.td2m_K - 273.15).astype(np.float64)
    t2_F = t2_C * 9.0 / 5.0 + 32.0

    # Compute RH
    rh = 100.0 * np.exp(17.625 * td2_C / (243.04 + td2_C)) / \
         np.exp(17.625 * t2_C / (243.04 + t2_C))
    rh = np.clip(rh, 0.0, 100.0)

    # Wind speed in mph
    wspd_ms = np.hypot(data.u10, data.v10).astype(np.float64)
    wspd_mph = wspd_ms * 2.237

    ffwi = _fosberg_v(t2_F, rh, wspd_mph)

    return {"ffwi": ffwi}


def compute_hot_dry_windy(data):
    """Compute Hot-Dry-Windy Index.

    Uses metrust hot_dry_windy(t_c, rh, wspd_ms, vpd).
    VPD computed as saturation_vapor_pressure(T) - vapor_pressure(Td).

    Returns dict with key: hdw (2D [ny, nx], unitless)
    """
    t2_C = (data.t2m_K - 273.15).astype(np.float64)
    td2_C = (data.td2m_K - 273.15).astype(np.float64)

    # Compute RH
    rh = 100.0 * np.exp(17.625 * td2_C / (243.04 + td2_C)) / \
         np.exp(17.625 * t2_C / (243.04 + t2_C))
    rh = np.clip(rh, 0.0, 100.0)

    wspd_ms = np.hypot(data.u10, data.v10).astype(np.float64)

    # VPD = saturation vapor pressure - actual vapor pressure (hPa)
    svp = np.array(_calc.saturation_vapor_pressure_array(t2_C.ravel())).reshape(t2_C.shape)
    vp = np.array(_calc.vapor_pressure_array(td2_C.ravel())).reshape(td2_C.shape)
    vpd = np.maximum(svp - vp, 0.0)

    hdw = _hot_dry_windy_v(t2_C, rh, wspd_ms, vpd)

    return {"hdw": hdw}


def compute_wbgt(data):
    """Compute Wet Bulb Globe Temperature (WBGT) in Fahrenheit.

    Requires DSWRF (downward shortwave radiation) to be available.
    Uses the Dimiceli globe temperature approximation.

    WBGT = 0.7 * Tw + 0.2 * Tg + 0.1 * Td (dry bulb)

    Returns dict with key: wbgt (2D [ny, nx] in F), or empty dict if
    DSWRF is unavailable.
    """
    if data.dswrf_wm2 is None:
        return {}

    if data.date is None:
        return {}

    ny, nx = data.ny, data.nx
    t2_C = (data.t2m_K - 273.15).astype(np.float64)
    td2_C = (data.td2m_K - 273.15).astype(np.float64)
    t2_K = data.t2m_K.astype(np.float64)

    # Wet bulb temperature (already have this computation in thermodynamic.py)
    psfc_hPa = (data.psfc_Pa / 100.0).ravel().astype(np.float64)
    wb = np.array(_calc.wet_bulb_temperature_array(
        psfc_hPa, t2_C.ravel().astype(np.float64), td2_C.ravel().astype(np.float64)
    )).reshape(ny, nx)

    # Wind speed in m/s and m/hr
    wspd_ms = np.hypot(data.u10, data.v10).astype(np.float64)
    wspd_m_hr = wspd_ms * 3600.0

    # Total incoming shortwave radiation
    S = data.dswrf_wm2.astype(np.float64)

    # Cloud cover fraction for direct/diffuse split
    if data.tcdc_pct is not None:
        cloud_frac = np.clip(data.tcdc_pct / 100.0, 0.0, 1.0)
    else:
        cloud_frac = np.zeros((ny, nx), dtype=np.float64)

    # Estimate direct and diffuse radiation from total and cloud cover
    # Simple split: direct fraction decreases with cloud cover
    f_dir = np.maximum(1.0 - cloud_frac, 0.0)
    f_dif_frac = 1.0 - f_dir

    # Solar zenith angle from lat/lon/datetime
    dt = data.date
    day_of_year = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0

    # Normalize longitudes to -180..180
    lons = data.lon.copy().astype(np.float64)
    lons[lons > 180] -= 360.0
    lats = data.lat.astype(np.float64)

    # Solar declination (Spencer, 1971)
    gamma = 2.0 * np.pi * (day_of_year - 1) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    # Equation of time (minutes)
    eqtime = (229.18 * (0.000075 + 0.001868 * np.cos(gamma)
              - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma)
              - 0.04089 * np.sin(2 * gamma)))

    # Solar hour angle
    time_offset = eqtime + 4.0 * lons  # minutes
    tst = hour_utc * 60.0 + time_offset  # true solar time in minutes
    ha = np.radians((tst / 4.0) - 180.0)  # hour angle in radians

    # Solar zenith angle
    lat_rad = np.radians(lats)
    cos_zen = (np.sin(lat_rad) * np.sin(decl) +
               np.cos(lat_rad) * np.cos(decl) * np.cos(ha))
    cos_zen = np.clip(cos_zen, 0.0, 1.0)  # below horizon = 0
    zen = np.arccos(cos_zen)

    # Stefan-Boltzmann constant
    sigma = 5.67e-8  # W/(m^2 K^4)

    # Vapor pressure from dewpoint
    e_a = np.array(_calc.vapor_pressure_array(
        td2_C.ravel().astype(np.float64)
    )).reshape(ny, nx)

    # Emissivity of atmosphere
    epsilon_a = 0.575 * np.power(np.maximum(e_a, 0.01), 1.0 / 7.0)

    # Direct beam and diffuse radiation components
    # Avoid division by zero at cos(zen) = 0
    cos_zen_safe = np.maximum(cos_zen, 0.01)

    f_db = S * f_dir          # direct beam irradiance [W/m^2]
    f_dif = S * f_dif_frac    # diffuse irradiance [W/m^2]

    # Globe temperature using Dimiceli formula
    # B = S * (f_db/(4*sigma*cos(z)) + 1.2/sigma * f_dif) + epsilon_a * (T_a+273.15)^4
    B = (f_db / (4.0 * sigma * cos_zen_safe) +
         1.2 / sigma * f_dif +
         epsilon_a * t2_K ** 4)

    # C = 0.315 * (wind_speed_m_per_hour)^0.58 / 5.3865e-8
    wspd_m_hr_safe = np.maximum(wspd_m_hr, 1.0)  # avoid zero wind
    C = 0.315 * np.power(wspd_m_hr_safe, 0.58) / 5.3865e-8

    # T_g = (B + C*T_a + 7680000) / (C + 256000)  in Celsius
    T_globe_C = (B + C * t2_C + 7680000.0) / (C + 256000.0)

    # WBGT = 0.7 * wet_bulb + 0.2 * T_globe + 0.1 * T_dry (all in C)
    wbgt_C = 0.7 * wb + 0.2 * T_globe_C + 0.1 * t2_C

    # Convert to Fahrenheit
    wbgt_F = wbgt_C * 9.0 / 5.0 + 32.0

    # Mask nighttime (solar zenith > ~90 degrees, cos_zen near zero)
    # For nighttime, use simplified indoor WBGT: 0.7*Tw + 0.3*Td
    night_mask = cos_zen < 0.01
    wbgt_indoor_C = 0.7 * wb + 0.3 * t2_C
    wbgt_indoor_F = wbgt_indoor_C * 9.0 / 5.0 + 32.0
    wbgt_F = np.where(night_mask, wbgt_indoor_F, wbgt_F)

    return {"wbgt": wbgt_F}
