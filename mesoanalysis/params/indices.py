"""Stability index computations using metrust engine.

Computes K-Index, Total Totals, Lifted Index, Boyden Index, and SWEAT Index.
All functions take an HRRRData (ModelData) dataclass and return dicts of 2D
[ny, nx] arrays.
"""

import numpy as np
from metrust._metrust import calc as _calc


def _extract_level(data_3d, levels_mb, target_mb):
    """Extract the 2D slice nearest to target_mb from a 3D array."""
    idx = np.argmin(np.abs(np.array(levels_mb) - target_mb))
    return data_3d[idx]


def _dewpoint_from_q(q_kgkg, p_hPa):
    """Compute dewpoint (C) from mixing ratio (kg/kg) and pressure (hPa)."""
    e = np.clip(q_kgkg * p_hPa / (0.622 + q_kgkg), 1e-6, None)
    td_C = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
    return td_C


def compute_stability_indices(data):
    """Compute K-Index, Total Totals, Boyden Index, and SWEAT Index.

    Extracts temperature and dewpoint at specific pressure levels from
    3D arrays, then applies metrust scalar functions element-wise.

    Returns dict with keys: k_index, total_totals, boyden, sweat
    (all 2D [ny, nx])
    """
    ny, nx = data.ny, data.nx
    levels = data.levels_mb

    # Extract temperature at key levels (Celsius)
    t_850 = _extract_level(data.t_C, levels, 850.0)
    t_700 = _extract_level(data.t_C, levels, 700.0)
    t_500 = _extract_level(data.t_C, levels, 500.0)

    # Extract dewpoint at key levels from mixing ratio
    q_850 = _extract_level(data.q_kgkg, levels, 850.0)
    q_700 = _extract_level(data.q_kgkg, levels, 700.0)

    td_850 = _dewpoint_from_q(q_850, 850.0)
    td_700 = _dewpoint_from_q(q_700, 700.0)

    # --- K-Index: (T850 - T500) + Td850 - (T700 - Td700) ---
    # Vectorized with numpy (formula is simple arithmetic)
    k_idx = (t_850 - t_500) + td_850 - (t_700 - td_700)

    # --- Total Totals: (T850 - T500) + (Td850 - T500) ---
    tt = (t_850 - t_500) + (td_850 - t_500)

    # --- Boyden Index: (Z700 - Z1000) / 10 - T700 - 200 ---
    # Need geopotential heights (MSL) at 700 and 1000 hPa
    h_700_agl = _extract_level(data.h_agl_m, levels, 700.0)
    # Convert AGL back to MSL for Boyden
    z_700 = h_700_agl + data.sfc_hgt

    # For 1000 hPa: check if we have it, otherwise extrapolate
    levels_arr = np.array(levels)
    if np.min(np.abs(levels_arr - 1000.0)) < 50:
        h_1000_agl = _extract_level(data.h_agl_m, levels, 1000.0)
        z_1000 = h_1000_agl + data.sfc_hgt
    else:
        # Approximate Z1000 from surface height and pressure
        # Use hypsometric equation: Z1000 ~ sfc_hgt - Rd*T/g * ln(psfc/1000)
        t_sfc_K = data.t2m_K
        psfc_hPa = data.psfc_Pa / 100.0
        Rd = 287.05
        g = 9.81
        z_1000 = data.sfc_hgt - (Rd * t_sfc_K / g) * np.log(psfc_hPa / 1000.0)

    # Vectorized Boyden
    boyden = (z_700 - z_1000) / 10.0 - t_700 - 200.0

    # --- SWEAT Index ---
    # Needs wind speed/direction at 850 and 500 hPa
    u_850 = _extract_level(data.u, levels, 850.0)
    v_850 = _extract_level(data.v, levels, 850.0)
    u_500 = _extract_level(data.u, levels, 500.0)
    v_500 = _extract_level(data.v, levels, 500.0)

    wspd_850_ms = np.hypot(u_850, v_850)
    wspd_500_ms = np.hypot(u_500, v_500)
    wspd_850_kt = wspd_850_ms * 1.944
    wspd_500_kt = wspd_500_ms * 1.944

    wdir_850 = (270.0 - np.degrees(np.arctan2(v_850, u_850))) % 360.0
    wdir_500 = (270.0 - np.degrees(np.arctan2(v_500, u_500))) % 360.0

    # SWEAT uses vectorized formula (metrust is scalar, so compute manually)
    # SWEAT = 12*Td850 + 20*TT_term + 2*wspd850 + wspd500 + 125*(sin(wdir500-wdir850)+0.2)
    # where TT_term = max(TT - 49, 0), Td850 term = max(Td850, 0)
    td850_term = np.maximum(td_850, 0.0) * 12.0
    tt_term = np.maximum(tt - 49.0, 0.0) * 20.0
    wspd850_term = wspd_850_kt * 2.0
    wspd500_term = wspd_500_kt

    # Shear term: only applied when specific wind direction criteria are met
    ddir = np.radians(wdir_500 - wdir_850)
    shear_term = 125.0 * (np.sin(ddir) + 0.2)

    # Wind direction criteria for shear term:
    # 130 <= wdir850 <= 250 AND 210 <= wdir500 <= 310
    # AND wdir500 - wdir850 > 0 AND wspd >= 15 kt
    dir_mask = (
        (wdir_850 >= 130) & (wdir_850 <= 250) &
        (wdir_500 >= 210) & (wdir_500 <= 310) &
        ((wdir_500 - wdir_850) > 0) &
        (wspd_850_kt >= 15) & (wspd_500_kt >= 15)
    )
    shear_term = np.where(dir_mask, shear_term, 0.0)

    sweat = td850_term + tt_term + wspd850_term + wspd500_term + shear_term
    sweat = np.maximum(sweat, 0.0)

    return {
        "k_index": k_idx,
        "total_totals": tt,
        "boyden": boyden,
        "sweat": sweat,
    }


def compute_lifted_index(data):
    """Compute Lifted Index per grid column.

    Lifts a surface parcel to 500 hPa. Uses metrust lifted_index()
    per column (profile-based function).

    Returns dict with key: lifted_index (2D [ny, nx])
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)
    levels = np.array(data.levels_mb, dtype=np.float64)

    # Surface parcel: use 2m T and Td
    t2_C = (data.t2m_K - 273.15).astype(np.float64)
    td2_C = (data.td2m_K - 273.15).astype(np.float64)
    psfc_hPa = (data.psfc_Pa / 100.0).astype(np.float64)

    # Build dewpoint 3D array from mixing ratio
    td_3d = np.empty_like(data.t_C)
    for k in range(nz):
        p_hPa = levels[k] if np.isscalar(data.p_Pa[k].flat[0]) else data.p_Pa[k] / 100.0
        if np.isscalar(p_hPa):
            td_3d[k] = _dewpoint_from_q(data.q_kgkg[k], p_hPa)
        else:
            td_3d[k] = _dewpoint_from_q(data.q_kgkg[k], p_hPa)

    li_out = np.full((ny, nx), np.nan, dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            # Build profile: surface + upper levels (surface first = highest pressure)
            p_col = np.empty(nz + 1, dtype=np.float64)
            t_col = np.empty(nz + 1, dtype=np.float64)
            td_col = np.empty(nz + 1, dtype=np.float64)

            p_col[0] = psfc_hPa[j, i]
            t_col[0] = t2_C[j, i]
            td_col[0] = td2_C[j, i]

            for k in range(nz):
                p_col[k + 1] = levels[k]
                t_col[k + 1] = data.t_C[k, j, i]
                td_col[k + 1] = td_3d[k, j, i]

            # Sort by pressure descending (surface first)
            sort_idx = np.argsort(-p_col)
            p_col = p_col[sort_idx]
            t_col = t_col[sort_idx]
            td_col = td_col[sort_idx]

            try:
                li_out[j, i] = _calc.lifted_index(p_col, t_col, td_col)
            except Exception:
                li_out[j, i] = np.nan

    return {"lifted_index": li_out}
