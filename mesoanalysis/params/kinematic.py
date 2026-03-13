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
