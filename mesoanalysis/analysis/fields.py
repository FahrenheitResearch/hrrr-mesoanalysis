"""
Merge Barnes-analyzed surface fields back into HRRR data.

After running Barnes objective analysis on individual surface variables,
this module replaces the corresponding fields in the HRRR dataset and
recomputes any derived quantities that depend on the corrected values.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

def _mixing_ratio_from_dewpoint(td_C: np.ndarray, p_hPa: np.ndarray) -> np.ndarray:
    """Compute mixing ratio (kg/kg) from dewpoint and pressure.

    Uses the Bolton (1980) approximation for saturation vapour pressure:

        e_s(T) = 6.112 * exp(17.67 * T / (T + 243.5))

    where T is in degrees Celsius and e_s is in hPa.

    Parameters
    ----------
    td_C : ndarray
        Dewpoint temperature (degrees Celsius).
    p_hPa : ndarray
        Pressure (hPa / mb).

    Returns
    -------
    q : ndarray
        Mixing ratio (kg/kg).
    """
    e = 6.112 * np.exp(17.67 * td_C / (td_C + 243.5))
    # Ensure vapour pressure doesn't exceed total pressure.
    e = np.minimum(e, p_hPa)
    q = 0.622 * e / (p_hPa - e)
    return np.maximum(q, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_analysis(
    hrrr_data: Any,
    analyzed_t2m: np.ndarray,
    analyzed_td2m: np.ndarray,
    analyzed_u10: np.ndarray,
    analyzed_v10: np.ndarray,
    analyzed_mslp: np.ndarray,
) -> Any:
    """Replace HRRR surface fields with Barnes-analyzed versions.

    Parameters
    ----------
    hrrr_data
        HRRR dataset object.  Expected to have attributes ``t2m`` (K),
        ``td2m`` (K), ``u10`` (m/s), ``v10`` (m/s), ``mslp`` (Pa or hPa),
        and ``surface_pressure`` (Pa or hPa).  If the object is a dataclass,
        ``dataclasses.replace`` is used; otherwise a shallow ``copy.copy``
        is made.
    analyzed_t2m : ndarray
        Analyzed 2-m temperature (degrees Celsius).
    analyzed_td2m : ndarray
        Analyzed 2-m dewpoint (degrees Celsius).
    analyzed_u10 : ndarray
        Analyzed 10-m u-wind component (m/s).
    analyzed_v10 : ndarray
        Analyzed 10-m v-wind component (m/s).
    analyzed_mslp : ndarray
        Analyzed mean sea-level pressure (hPa).

    Returns
    -------
    modified : same type as *hrrr_data*
        A copy of *hrrr_data* with corrected surface fields and
        recomputed surface mixing ratio.
    """
    import dataclasses

    logger.info("Merging analyzed surface fields into HRRR data.")

    # -- compute derived fields before building the output -------------------
    # Determine surface pressure for mixing-ratio calculation.
    sfc_p = getattr(hrrr_data, "psfc_Pa", None)
    if sfc_p is None:
        logger.debug(
            "No psfc_Pa attribute; using analyzed MSLP for "
            "mixing-ratio calculation."
        )
        sfc_p = analyzed_mslp

    # Normalise pressure to hPa (if stored in Pa, values will be > 10000).
    sfc_p = np.asarray(sfc_p, dtype=np.float64)
    if np.nanmean(sfc_p) > 10_000:
        sfc_p_hPa = sfc_p / 100.0
    else:
        sfc_p_hPa = sfc_p

    q_sfc = _mixing_ratio_from_dewpoint(analyzed_td2m, sfc_p_hPa)

    # -- build replacement field dict ----------------------------------------
    new_fields = {
        "t2m_K": analyzed_t2m + 273.15,       # C -> K
        "td2m_K": analyzed_td2m + 273.15,     # C -> K
        "u10": analyzed_u10.copy(),
        "v10": analyzed_v10.copy(),
        "mslp_Pa": analyzed_mslp * 100.0,     # hPa -> Pa
    }

    # -- create a modified copy ----------------------------------------------
    is_dc = dataclasses.is_dataclass(hrrr_data) and not isinstance(hrrr_data, type)

    if is_dc:
        # Only pass fields that actually exist on the dataclass.
        dc_names = {f.name for f in dataclasses.fields(hrrr_data)}
        replace_kwargs = {k: v for k, v in new_fields.items() if k in dc_names}
        out = dataclasses.replace(hrrr_data, **replace_kwargs)
    else:
        out = copy.copy(hrrr_data)
        for attr, value in new_fields.items():
            setattr(out, attr, value)

    logger.info(
        "Surface merge complete. t2m range [%.1f, %.1f] K",
        float(np.nanmin(new_fields["t2m_K"])),
        float(np.nanmax(new_fields["t2m_K"])),
    )

    return out
