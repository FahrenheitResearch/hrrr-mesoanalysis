"""Kinematic parameter computations using metrust engine.

All functions take an HRRRData dataclass and return dicts of 2D [ny, nx] arrays.
"""

import numpy as np
from metrust._metrust import calc as _calc

# Conversion factor: m/s to knots
MS_TO_KT = 1.944


def compute_shear_fields(data):
    """Compute bulk wind shear at various depths.

    Uses _calc.compute_shear(u_3d, v_3d, h_3d, nx, ny, nz, bot_m, top_m)
    which returns (u_shear, v_shear) as two 1D arrays of length nx*ny.

    Computes 0-1km, 0-3km, 0-6km bulk shear magnitude in knots.

    Returns dict with keys:
        shear_01km, shear_03km, shear_06km: 2D [ny, nx] in knots
        shear_06km_ms: 2D [ny, nx] in m/s (for composite calculations)
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)

    u_3d = data.u.ravel().astype(np.float64)
    v_3d = data.v.ravel().astype(np.float64)
    h_3d = data.h_agl_m.ravel().astype(np.float64)

    result = {}

    for label, bot, top in [("01km", 0.0, 1000.0),
                             ("03km", 0.0, 3000.0),
                             ("06km", 0.0, 6000.0)]:
        shear_mag = np.array(_calc.compute_shear(u_3d, v_3d, h_3d,
                                                  nx, ny, nz, bot, top))
        mag_ms = shear_mag.reshape(ny, nx)
        result[f"shear_{label}"] = mag_ms * MS_TO_KT

        if label == "06km":
            result["shear_06km_ms"] = mag_ms

    return result


def compute_srh_fields(data):
    """Compute storm-relative helicity.

    Uses _calc.compute_srh(u_3d, v_3d, h_3d, nx, ny, nz, depth_m)
    which returns a 1D array of length nx*ny.

    Computes 0-1km and 0-3km SRH.

    Returns dict with keys: srh_01km, srh_03km (2D [ny, nx] in m^2/s^2)
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)

    u_3d = data.u.ravel().astype(np.float64)
    v_3d = data.v.ravel().astype(np.float64)
    h_3d = data.h_agl_m.ravel().astype(np.float64)

    srh_01 = np.array(_calc.compute_srh(u_3d, v_3d, h_3d,
                                         nx, ny, nz, 1000.0)).reshape(ny, nx)
    srh_03 = np.array(_calc.compute_srh(u_3d, v_3d, h_3d,
                                         nx, ny, nz, 3000.0)).reshape(ny, nx)

    return {"srh_01km": srh_01, "srh_03km": srh_03}


def compute_bunkers_storm_motion(data):
    """Compute Bunkers storm motion per grid column.

    Uses metrust bunkers_storm_motion(p_prof, u_prof, v_prof, height_prof)
    which returns ((rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v)).

    Returns dict with keys:
        bunkers_rm_u, bunkers_rm_v: right-mover storm motion u/v [m/s]
        bunkers_lm_u, bunkers_lm_v: left-mover storm motion u/v [m/s]
    (all 2D [ny, nx])
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)
    levels = np.array(data.levels_mb, dtype=np.float64)

    rm_u = np.zeros((ny, nx), dtype=np.float64)
    rm_v = np.zeros((ny, nx), dtype=np.float64)
    lm_u = np.zeros((ny, nx), dtype=np.float64)
    lm_v = np.zeros((ny, nx), dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            p_col = levels.copy()
            u_col = np.array([data.u[k, j, i] for k in range(nz)], dtype=np.float64)
            v_col = np.array([data.v[k, j, i] for k in range(nz)], dtype=np.float64)
            h_col = np.array([data.h_agl_m[k, j, i] for k in range(nz)], dtype=np.float64)

            # Sort by pressure descending (surface first)
            sort_idx = np.argsort(-p_col)
            p_col = p_col[sort_idx]
            u_col = u_col[sort_idx]
            v_col = v_col[sort_idx]
            h_col = h_col[sort_idx]

            try:
                result = _calc.bunkers_storm_motion(p_col, u_col, v_col, h_col)
                # result = ((rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v))
                rm_u[j, i] = result[0][0]
                rm_v[j, i] = result[0][1]
                lm_u[j, i] = result[1][0]
                lm_v[j, i] = result[1][1]
            except Exception:
                rm_u[j, i] = np.nan
                rm_v[j, i] = np.nan
                lm_u[j, i] = np.nan
                lm_v[j, i] = np.nan

    return {
        "bunkers_rm_u": rm_u, "bunkers_rm_v": rm_v,
        "bunkers_lm_u": lm_u, "bunkers_lm_v": lm_v,
    }
