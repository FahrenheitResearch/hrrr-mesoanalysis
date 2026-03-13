"""Composite severe weather parameter computations.

Combines thermodynamic and kinematic fields into composite indices.
"""

import numpy as np
from metrust._metrust import calc as _calc


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
