"""Composite severe weather parameter computations.

Combines thermodynamic and kinematic fields into composite indices.
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


def compute_composite_fields(data, thermo, kinematic):
    """Compute composite severe weather parameters.

    Args:
        data: HRRRData dataclass
        thermo: merged dict from compute_cape_fields, compute_theta_e, etc.
        kinematic: merged dict from compute_shear_fields, compute_srh_fields

    Returns dict with keys: stp, scp (2D [ny, nx])
    """
    sbcape = thermo["sbcape"]
    mucape = thermo["mucape"]
    lcl_m = thermo.get("lcl_m")
    srh_01km = kinematic["srh_01km"]
    srh_03km = kinematic["srh_03km"]
    shear_06km_ms = kinematic["shear_06km_ms"]

    # --- STP (Significant Tornado Parameter) ---
    try:
        sbcape_1d = sbcape.ravel().astype(np.float64)
        lcl_1d = lcl_m.ravel().astype(np.float64) if lcl_m is not None else None
        srh1_1d = srh_01km.ravel().astype(np.float64)
        shear6_1d = (shear_06km_ms * 1.944).ravel().astype(np.float64)  # kt
        stp_flat = np.array(_calc.compute_stp(sbcape_1d, lcl_1d,
                                               srh1_1d, shear6_1d))
        stp = stp_flat.reshape(data.ny, data.nx)
    except (AttributeError, TypeError):
        # Manual STP calculation
        cape_term = sbcape / 1500.0
        cape_term = np.clip(cape_term, 0.0, None)

        if lcl_m is not None:
            lcl_term = (2000.0 - lcl_m) / 1000.0
            lcl_term = np.clip(lcl_term, 0.0, None)
        else:
            lcl_term = np.ones_like(sbcape)

        srh_term = srh_01km / 150.0
        srh_term = np.clip(srh_term, 0.0, None)

        shear_term = shear_06km_ms / 20.0
        shear_term = np.clip(shear_term, 0.0, None)

        stp = cape_term * lcl_term * srh_term * shear_term

    # --- SCP (Supercell Composite Parameter) ---
    mucape_term = mucape / 1000.0
    mucape_term = np.clip(mucape_term, 0.0, None)

    srh3_term = srh_03km / 50.0
    srh3_term = np.clip(srh3_term, 0.0, None)

    shear6_term = shear_06km_ms / 20.0
    shear6_term = np.clip(shear6_term, 0.0, None)

    scp = mucape_term * srh3_term * shear6_term

    return {"stp": stp, "scp": scp}


def compute_ehi(thermo, kinematic):
    """Compute Energy-Helicity Index.

    EHI = (SBCAPE * SRH_01km) / 160000

    Uses metrust compute_ehi(cape, srh) on flattened arrays.

    Returns dict with key: ehi (2D [ny, nx])
    """
    sbcape = thermo["sbcape"]
    srh_01km = kinematic["srh_01km"]
    ny, nx = sbcape.shape

    cape_1d = sbcape.ravel().astype(np.float64)
    srh_1d = srh_01km.ravel().astype(np.float64)

    ehi_flat = np.array(_calc.compute_ehi(cape_1d, srh_1d))
    ehi = ehi_flat.reshape(ny, nx)

    return {"ehi": ehi}


def compute_ship(data, thermo, kinematic):
    """Compute Significant Hail Parameter (SHIP).

    Uses metrust significant_hail_parameter(cape, shear06, t500,
    lr_700_500, mr, nx, ny) on flattened arrays.

    Returns dict with key: ship (2D [ny, nx])
    """
    ny, nx = data.ny, data.nx

    mucape_1d = thermo["mucape"].ravel().astype(np.float64)
    shear06_1d = kinematic["shear_06km_ms"].ravel().astype(np.float64)
    lr_1d = thermo["lr_700_500"].ravel().astype(np.float64)
    mr_1d = thermo["mixing_ratio"].ravel().astype(np.float64)

    # Extract T at 500 hPa
    t_500 = _extract_level(data.t_C, data.levels_mb, 500.0)
    t500_1d = t_500.ravel().astype(np.float64)

    ship_flat = np.array(_calc.significant_hail_parameter(
        mucape_1d, shear06_1d, t500_1d, lr_1d, mr_1d, nx, ny
    ))
    ship = ship_flat.reshape(ny, nx)

    return {"ship": ship}


def compute_dcape(data):
    """Compute Downdraft CAPE per grid column.

    Uses metrust downdraft_cape(pressure, temperature, dewpoint) per column.
    Profiles are surface-first (decreasing pressure), with p in hPa,
    t and td in Celsius.

    Profiles are pre-sorted once and columns extracted via numpy slicing
    to minimise per-column Python overhead.

    Returns dict with key: dcape (2D [ny, nx] in J/kg)
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)
    levels = np.array(data.levels_mb, dtype=np.float64)

    # Build dewpoint 3D array from mixing ratio (vectorised per level)
    td_3d = np.empty_like(data.t_C)
    for k in range(nz):
        td_3d[k] = _dewpoint_from_q(data.q_kgkg[k], levels[k])

    # Pre-sort by pressure descending (surface first) once for all columns
    sort_idx = np.argsort(-levels)
    p_sorted = levels[sort_idx]
    t_sorted = data.t_C[sort_idx].astype(np.float64)
    td_sorted = td_3d[sort_idx].astype(np.float64)

    dcape_out = np.zeros((ny, nx), dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            t_col = t_sorted[:, j, i].copy()
            td_col = td_sorted[:, j, i].copy()

            try:
                dcape_out[j, i] = _calc.downdraft_cape(p_sorted, t_col, td_col)
            except Exception:
                dcape_out[j, i] = 0.0

    return {"dcape": dcape_out}


def compute_dcp(data, thermo, kinematic, dcape_field):
    """Compute Derecho Composite Parameter (DCP).

    DCP = (DCAPE/980) * (MUCAPE/2000) * (SHEAR_06/20) * (MU_MR/11)

    Uses metrust derecho_composite_parameter(dcape, mu_cape, shear06,
    mu_mixing_ratio, nx, ny).

    Returns dict with key: dcp (2D [ny, nx])
    """
    ny, nx = data.ny, data.nx

    dcape_1d = dcape_field.ravel().astype(np.float64)
    mucape_1d = thermo["mucape"].ravel().astype(np.float64)
    shear06_1d = kinematic["shear_06km_ms"].ravel().astype(np.float64)
    mr_1d = thermo["mixing_ratio"].ravel().astype(np.float64)

    dcp_flat = np.array(_calc.derecho_composite_parameter(
        dcape_1d, mucape_1d, shear06_1d, mr_1d, nx, ny
    ))
    dcp = dcp_flat.reshape(ny, nx)

    return {"dcp": dcp}


def compute_enhanced_scp(data, thermo, kinematic):
    """Compute Enhanced Supercell Composite Parameter.

    SCP = (MUCAPE/1000) * (SRH/50) * (SHEAR_06/40) * CIN_term

    Uses metrust grid_supercell_composite_parameter(mu_cape, srh,
    shear_06, mu_cin, nx, ny).

    Returns dict with key: scp_enhanced (2D [ny, nx])
    """
    ny, nx = data.ny, data.nx

    mucape_1d = thermo["mucape"].ravel().astype(np.float64)
    srh_1d = kinematic["srh_03km"].ravel().astype(np.float64)
    shear06_1d = kinematic["shear_06km_ms"].ravel().astype(np.float64)
    mucin_1d = thermo["mucin"].ravel().astype(np.float64)

    scp_flat = np.array(_calc.grid_supercell_composite_parameter(
        mucape_1d, srh_1d, shear06_1d, mucin_1d, nx, ny
    ))
    scp_enh = scp_flat.reshape(ny, nx)

    return {"scp_enhanced": scp_enh}


def compute_critical_angle(data, kinematic):
    """Compute Critical Angle on 2D grids.

    Uses metrust grid_critical_angle(u_storm, v_storm, u_shear, v_shear, nx, ny).
    Needs Bunkers storm motion (right mover) and 0-1km shear components.

    Profiles are pre-sorted by height ascending once, then columns are
    extracted via numpy slicing.

    Returns dict with key: critical_angle (2D [ny, nx] in degrees)
    """
    ny, nx = data.ny, data.nx
    nz = len(data.levels_mb)

    # Get Bunkers right-mover storm motion
    bunkers_rm_u = kinematic.get("bunkers_rm_u")
    bunkers_rm_v = kinematic.get("bunkers_rm_v")
    if bunkers_rm_u is None or bunkers_rm_v is None:
        return {}

    # Compute 0-1km shear u/v components per column using bulk_shear
    # Pre-sort by height ascending using a representative column (column 0,0)
    # Since pressure levels are fixed, the height ordering is consistent
    # across columns (higher pressure = lower height).
    # Sort once based on the level ordering.
    h_mean = np.mean(data.h_agl_m, axis=(1, 2))  # mean height per level
    sort_idx = np.argsort(h_mean)
    u_sorted = data.u[sort_idx].astype(np.float64)
    v_sorted = data.v[sort_idx].astype(np.float64)
    h_sorted = data.h_agl_m[sort_idx].astype(np.float64)

    u_shear_01 = np.zeros((ny, nx), dtype=np.float64)
    v_shear_01 = np.zeros((ny, nx), dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            u_col = u_sorted[:, j, i].copy()
            v_col = v_sorted[:, j, i].copy()
            h_col = h_sorted[:, j, i].copy()

            try:
                us, vs = _calc.bulk_shear(u_col, v_col, h_col, 0.0, 1000.0)
                u_shear_01[j, i] = us
                v_shear_01[j, i] = vs
            except Exception:
                u_shear_01[j, i] = 0.0
                v_shear_01[j, i] = 0.0

    u_storm_1d = bunkers_rm_u.ravel().astype(np.float64)
    v_storm_1d = bunkers_rm_v.ravel().astype(np.float64)
    u_shear_1d = u_shear_01.ravel().astype(np.float64)
    v_shear_1d = v_shear_01.ravel().astype(np.float64)

    ca_flat = np.array(_calc.grid_critical_angle(
        u_storm_1d, v_storm_1d, u_shear_1d, v_shear_1d, nx, ny
    ))
    critical_angle = ca_flat.reshape(ny, nx)

    return {"critical_angle": critical_angle}
