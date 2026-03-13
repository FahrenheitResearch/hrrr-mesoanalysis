"""Thermodynamic parameter computations using metrust engine.

All functions take an HRRRData dataclass and return 2D [ny, nx] arrays
or dicts of 2D arrays.
"""

import numpy as np
from metrust._metrust import calc as _calc


def compute_cape_fields(data):
    """Compute SBCAPE, MLCAPE, MUCAPE, SB3CAPE, LCL heights.

    Uses _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2,
                                 nx, ny, nz, parcel_type, top_m=None)

    Units expected by compute_cape_cin:
    - p_3d: Pa (divided by 100 internally)
    - t_3d: Celsius
    - q_3d: kg/kg mixing ratio
    - h_3d: meters AGL
    - psfc: Pa
    - t2: Kelvin (subtracts 273.15 internally)
    - q2: kg/kg
    - All arrays must be flattened, in [nz][ny][nx] order for 3D, [ny][nx] for 2D

    Returns dict with keys: sbcape, sbcin, mlcape, mlcin, mucape, mucin,
                            sb3cape, lcl_p
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)

    # Flatten 3D arrays (already in [nz, ny, nx] order)
    p_3d = data.p_Pa.ravel().astype(np.float64)
    t_3d = data.t_C.ravel().astype(np.float64)
    q_3d = data.q_kgkg.ravel().astype(np.float64)
    h_3d = data.h_agl_m.ravel().astype(np.float64)

    # Flatten 2D surface arrays
    psfc = data.psfc_Pa.ravel().astype(np.float64)
    t2 = data.t2m_K.ravel().astype(np.float64)

    # Compute surface mixing ratio from dewpoint for q2
    td2_C = data.td2m_K - 273.15
    e_sat = 6.112 * np.exp(17.67 * td2_C / (td2_C + 243.5))
    psfc_hPa = data.psfc_Pa / 100.0
    q2 = (0.622 * e_sat / (psfc_hPa - e_sat)).ravel().astype(np.float64)

    result = {}

    # Surface-based parcel
    sb = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2,
                                nx, ny, nz, "surface")
    result["sbcape"] = np.array(sb[0]).reshape(ny, nx)
    result["sbcin"] = np.array(sb[1]).reshape(ny, nx)
    result["lcl_p"] = np.array(sb[2]).reshape(ny, nx) if len(sb) > 2 else None

    # Mixed-layer parcel
    ml = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2,
                                nx, ny, nz, "mixed_layer")
    result["mlcape"] = np.array(ml[0]).reshape(ny, nx)
    result["mlcin"] = np.array(ml[1]).reshape(ny, nx)

    # Most-unstable parcel
    mu = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2,
                                nx, ny, nz, "most_unstable")
    result["mucape"] = np.array(mu[0]).reshape(ny, nx)
    result["mucin"] = np.array(mu[1]).reshape(ny, nx)

    # Surface-based CAPE in 0-3km layer
    sb3 = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2,
                                 nx, ny, nz, "surface", top_m=3000.0)
    result["sb3cape"] = np.array(sb3[0]).reshape(ny, nx)

    # Convert LCL pressure to height (m AGL) if lcl_p is available
    if result["lcl_p"] is not None:
        # Use hypsometric approximation: lcl_m ~ (1 - (lcl_p/psfc)^0.19) * T2/0.0065
        t2_2d = data.t2m_K
        psfc_2d = data.psfc_Pa
        lcl_p_2d = result["lcl_p"]
        with np.errstate(invalid="ignore"):
            ratio = np.clip(lcl_p_2d / psfc_2d, 0.0, 1.0)
            result["lcl_m"] = (1.0 - ratio ** 0.190284) * t2_2d / 0.0065

    return result


def compute_theta_e(data):
    """Compute surface and 850mb equivalent potential temperature.

    Uses _calc.equivalent_potential_temperature_array(p_hPa, t_C, td_C)
    which returns an array of theta-e in Kelvin.

    Returns dict with keys: theta_e_sfc, theta_e_850 (2D [ny, nx] in K)
    """
    ny, nx = data.ny, data.nx

    # Surface theta-e
    t2_C = (data.t2m_K - 273.15).ravel().astype(np.float64)
    td2_C = (data.td2m_K - 273.15).ravel().astype(np.float64)
    psfc_hPa = (data.psfc_Pa / 100.0).ravel().astype(np.float64)

    te_sfc = np.array(_calc.equivalent_potential_temperature_array(
        psfc_hPa, t2_C, td2_C
    )).reshape(ny, nx)

    # 850mb theta-e: find 850mb level index
    levels = np.array(data.levels_mb)
    idx_850 = np.argmin(np.abs(levels - 850.0))

    t_850_C = data.t_C[idx_850].ravel().astype(np.float64)
    # Compute dewpoint from mixing ratio at 850mb
    q_850 = data.q_kgkg[idx_850]
    p_850_hPa = data.p_Pa[idx_850] / 100.0
    e = q_850 * p_850_hPa / (0.622 + q_850)
    e = np.clip(e, 1e-6, None)
    td_850_C = (243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112)))
    td_850_C = td_850_C.ravel().astype(np.float64)
    p_850_flat = p_850_hPa.ravel().astype(np.float64)

    te_850 = np.array(_calc.equivalent_potential_temperature_array(
        p_850_flat, t_850_C, td_850_C
    )).reshape(ny, nx)

    return {"theta_e_sfc": te_sfc, "theta_e_850": te_850}


def compute_wet_bulb(data):
    """Compute surface wet bulb temperature.

    Uses _calc.wet_bulb_temperature_array(p_hPa, t_C, td_C) returning Celsius.

    Returns dict with key: wetbulb_sfc (2D [ny, nx] in C)
    """
    ny, nx = data.ny, data.nx

    t2_C = (data.t2m_K - 273.15).ravel().astype(np.float64)
    td2_C = (data.td2m_K - 273.15).ravel().astype(np.float64)
    psfc_hPa = (data.psfc_Pa / 100.0).ravel().astype(np.float64)

    wb = np.array(_calc.wet_bulb_temperature_array(
        psfc_hPa, t2_C, td2_C
    )).reshape(ny, nx)

    return {"wetbulb_sfc": wb}


def compute_lapse_rates(data):
    """Compute 700-500mb and 850-500mb lapse rates.

    Interpolates temperature and height to exact pressure levels from
    the 3D arrays, then computes -dT/dz in C/km.

    Returns dict with keys: lr_700_500, lr_850_500 (2D [ny, nx] in C/km)
    """
    ny, nx = data.ny, data.nx
    levels = np.array(data.levels_mb, dtype=np.float64)

    def _interp_level(field_3d, target_mb):
        """Linearly interpolate a 3D field to a target pressure level."""
        # Find bounding levels (levels are typically top-down or bottom-up)
        # Work in log-pressure space for accuracy
        log_levels = np.log(levels)
        log_target = np.log(target_mb)

        # Find the two bounding indices
        idx_below = np.searchsorted(-log_levels, -log_target) - 1
        idx_below = np.clip(idx_below, 0, len(levels) - 2)
        idx_above = idx_below + 1

        log_lo = log_levels[idx_below]
        log_hi = log_levels[idx_above]
        weight = (log_target - log_lo) / (log_hi - log_lo) if log_hi != log_lo else 0.0

        return field_3d[idx_below] * (1.0 - weight) + field_3d[idx_above] * weight

    t_500 = _interp_level(data.t_C, 500.0)
    t_700 = _interp_level(data.t_C, 700.0)
    t_850 = _interp_level(data.t_C, 850.0)

    h_500 = _interp_level(data.h_agl_m, 500.0)
    h_700 = _interp_level(data.h_agl_m, 700.0)
    h_850 = _interp_level(data.h_agl_m, 850.0)

    # Lapse rate = -dT/dz in C/km (positive means temperature decreasing with height)
    dz_700_500 = (h_500 - h_700) / 1000.0  # km
    dz_850_500 = (h_500 - h_850) / 1000.0  # km

    lr_700_500 = np.where(dz_700_500 > 0, -(t_500 - t_700) / dz_700_500, 0.0)
    lr_850_500 = np.where(dz_850_500 > 0, -(t_500 - t_850) / dz_850_500, 0.0)

    return {"lr_700_500": lr_700_500, "lr_850_500": lr_850_500}


def compute_mixing_ratio(data):
    """Compute surface mixing ratio in g/kg from t2m and td2m.

    Returns dict with key: mixr_sfc (2D [ny, nx] in g/kg)
    """
    td2_C = data.td2m_K - 273.15
    psfc_hPa = data.psfc_Pa / 100.0

    # Saturation vapor pressure at dewpoint (Bolton 1980)
    e = 6.112 * np.exp(17.67 * td2_C / (td2_C + 243.5))
    mixr = 622.0 * e / (psfc_hPa - e)  # g/kg

    return {"mixing_ratio": mixr}


def compute_pw(data):
    """Compute precipitable water in mm (kg/m^2).

    Uses _calc.compute_pw(q_3d, p_3d, nx, ny, nz) if available,
    otherwise integrates manually: PW = -(1/g) * integral(q dp).

    Returns dict with key: pw (2D [ny, nx] in mm)
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)

    q_3d = data.q_kgkg.ravel().astype(np.float64)
    p_3d = data.p_Pa.ravel().astype(np.float64)

    try:
        pw_flat = np.array(_calc.compute_pw(q_3d, p_3d, nx, ny, nz))
        pw = pw_flat.reshape(ny, nx)
    except (AttributeError, TypeError):
        # Manual integration: PW = (1/g) * sum(q * dp) through column
        g = 9.81
        pw = np.zeros((ny, nx), dtype=np.float64)
        for k in range(nz - 1):
            q_avg = 0.5 * (data.q_kgkg[k] + data.q_kgkg[k + 1])
            dp = np.abs(data.p_Pa[k] - data.p_Pa[k + 1])
            pw += q_avg * dp / g  # kg/m^2 = mm

    return {"pw": pw}
