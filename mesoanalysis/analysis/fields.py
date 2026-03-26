"""
Merge Barnes-analyzed surface fields back into model data.

After running Barnes objective analysis on individual surface variables,
this module replaces the corresponding fields in the model dataset and
recomputes any derived quantities that depend on the corrected values.

Boundary-layer vertical blending ensures that surface corrections taper
smoothly into the lowest pressure levels of the 3-D profile, preventing
inconsistencies when computing CAPE/CIN/LCL from a corrected surface
parcel rising through an uncorrected low-level environment.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from mesoanalysis.models import ModelData

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


def _dewpoint_from_mixing_ratio(q_kgkg: np.ndarray, p_hPa: np.ndarray) -> np.ndarray:
    """Compute dewpoint (C) from mixing ratio and pressure.

    Inverts the Bolton (1980) formula:
        e = q * p / (0.622 + q)
        td = 243.5 * ln(e/6.112) / (17.67 - ln(e/6.112))

    Parameters
    ----------
    q_kgkg : ndarray
        Mixing ratio (kg/kg).
    p_hPa : ndarray
        Pressure (hPa / mb).

    Returns
    -------
    td_C : ndarray
        Dewpoint temperature (degrees Celsius).
    """
    q_safe = np.maximum(q_kgkg, 1e-10)
    e = q_safe * p_hPa / (0.622 + q_safe)
    e = np.maximum(e, 1e-10)  # guard against log(0)
    ln_ratio = np.log(e / 6.112)
    td_C = 243.5 * ln_ratio / (17.67 - ln_ratio)
    return td_C


# ---------------------------------------------------------------------------
# Boundary-layer vertical blending
# ---------------------------------------------------------------------------

def _blend_bl_corrections(
    model_data: ModelData,
    delta_t: np.ndarray,
    delta_td: np.ndarray,
    delta_u: np.ndarray,
    delta_v: np.ndarray,
    psfc_hPa: np.ndarray,
    bl_taper_hpa: float,
) -> None:
    """Apply exponentially tapered surface corrections into the 3-D profile.

    Modifies ``model_data.t_C``, ``model_data.q_kgkg``, ``model_data.u``,
    and ``model_data.v`` **in place**.

    Parameters
    ----------
    model_data : ModelData
        Model dataset whose 3-D arrays will be modified in place.
    delta_t : ndarray (ny, nx)
        Surface temperature correction (C).
    delta_td : ndarray (ny, nx)
        Surface dewpoint correction (C).
    delta_u, delta_v : ndarray (ny, nx)
        Surface wind corrections (m/s).
    psfc_hPa : ndarray (ny, nx)
        Surface pressure (hPa).
    bl_taper_hpa : float
        Gaussian e-folding scale height for the taper (hPa).
    """
    levels = np.asarray(model_data.levels_mb, dtype=np.float64)
    min_level_hPa = 850.0  # safety cutoff — don't touch free troposphere

    # Identify levels eligible for blending (>= 850 hPa).
    eligible = levels >= min_level_hPa
    eligible_indices = np.where(eligible)[0]

    if len(eligible_indices) == 0:
        logger.info("BL taper: no levels >= %.0f hPa; skipping.", min_level_hPa)
        return

    n_blended = 0
    for k in eligible_indices:
        p_lev = levels[k]  # scalar pressure of this level (hPa)

        # Gaussian weight: 1 at surface, decays with distance above surface.
        # psfc_hPa is (ny, nx); p_lev is scalar.
        dp = psfc_hPa - p_lev  # positive when level is above surface
        weight = np.exp(-((dp / bl_taper_hpa) ** 2))

        # Where the level is below the surface (dp < 0, underground), still
        # allow weight = 1 (same as surface).  The Gaussian handles this
        # naturally since dp<0 => weight ~ 1 for small |dp|.

        # Where psfc < p_lev the level is underground — skip those points
        # by zeroing their weight so we don't introduce artefacts.
        weight = np.where(dp < -10.0, 0.0, weight)

        # Skip this level entirely if all weights are negligible.
        if np.nanmax(weight) < 1e-6:
            continue

        n_blended += 1

        # --- Temperature ---
        model_data.t_C[k] = model_data.t_C[k] + delta_t * weight

        # --- Wind ---
        model_data.u[k] = model_data.u[k] + delta_u * weight
        model_data.v[k] = model_data.v[k] + delta_v * weight

        # --- Moisture (dewpoint-based to preserve physical consistency) ---
        # Convert current q at this level back to dewpoint.
        p_lev_arr = np.full_like(psfc_hPa, p_lev)
        td_orig = _dewpoint_from_mixing_ratio(model_data.q_kgkg[k], p_lev_arr)

        # Apply tapered dewpoint correction.
        td_tapered = td_orig + delta_td * weight

        # Convert back to mixing ratio.
        model_data.q_kgkg[k] = _mixing_ratio_from_dewpoint(td_tapered, p_lev_arr)

    logger.info(
        "BL taper: blending corrections into lowest %d levels "
        "(scale=%.0f hPa, cutoff=%.0f hPa).",
        n_blended, bl_taper_hpa, min_level_hPa,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_analysis(
    model_data: ModelData,
    analyzed_t2m: np.ndarray,
    analyzed_td2m: np.ndarray,
    analyzed_u10: np.ndarray,
    analyzed_v10: np.ndarray,
    analyzed_mslp: np.ndarray,
    bl_taper_hpa: float = 150.0,
) -> ModelData:
    """Replace model surface fields with Barnes-analyzed versions.

    After replacing 2-m / 10-m fields, exponentially tapered corrections
    are blended into the lowest pressure levels of the 3-D profile so
    that CAPE / CIN / LCL computations see a consistent boundary layer.

    Parameters
    ----------
    model_data : ModelData
        Model dataset object with attributes ``t2m_K`` (K),
        ``td2m_K`` (K), ``u10`` (m/s), ``v10`` (m/s), ``mslp_Pa`` (Pa),
        and ``psfc_Pa`` (Pa).
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
    bl_taper_hpa : float, optional
        Gaussian e-folding scale height (hPa) for the boundary-layer
        blending taper.  Default 150 hPa (corrections decay to ~zero by
        850 hPa for a 1000 hPa surface).

    Returns
    -------
    modified : ModelData
        A copy of *model_data* with corrected surface fields,
        recomputed surface mixing ratio, and blended boundary-layer
        profiles.
    """
    import dataclasses

    logger.info("Merging analyzed surface fields into model data.")

    # -- save original surface values BEFORE replacement --------------------
    orig_t2m_C = model_data.t2m_K - 273.15
    orig_td2m_C = model_data.td2m_K - 273.15
    orig_u10 = model_data.u10.copy()
    orig_v10 = model_data.v10.copy()

    # -- compute derived fields before building the output -------------------
    # Determine surface pressure for mixing-ratio calculation.
    sfc_p = getattr(model_data, "psfc_Pa", None)
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
    is_dc = dataclasses.is_dataclass(model_data) and not isinstance(model_data, type)

    if is_dc:
        # Only pass fields that actually exist on the dataclass.
        dc_names = {f.name for f in dataclasses.fields(model_data)}
        replace_kwargs = {k: v for k, v in new_fields.items() if k in dc_names}
        out = dataclasses.replace(model_data, **replace_kwargs)
    else:
        out = copy.copy(model_data)
        for attr, value in new_fields.items():
            setattr(out, attr, value)

    logger.info(
        "Surface merge complete. t2m range [%.1f, %.1f] K",
        float(np.nanmin(new_fields["t2m_K"])),
        float(np.nanmax(new_fields["t2m_K"])),
    )

    # -- boundary-layer vertical blending ------------------------------------
    has_3d = (
        getattr(out, "t_C", None) is not None
        and getattr(out, "q_kgkg", None) is not None
        and getattr(out, "u", None) is not None
        and getattr(out, "v", None) is not None
        and getattr(out, "levels_mb", None) is not None
        and len(out.levels_mb) > 0
    )

    if has_3d and bl_taper_hpa > 0:
        # Surface correction increments.
        delta_t = analyzed_t2m - orig_t2m_C
        delta_td = analyzed_td2m - orig_td2m_C
        delta_u = analyzed_u10 - orig_u10
        delta_v = analyzed_v10 - orig_v10

        _blend_bl_corrections(
            out,
            delta_t=delta_t,
            delta_td=delta_td,
            delta_u=delta_u,
            delta_v=delta_v,
            psfc_hPa=sfc_p_hPa,
            bl_taper_hpa=bl_taper_hpa,
        )
    elif bl_taper_hpa > 0:
        logger.debug(
            "BL taper requested but 3-D fields not available; skipping."
        )

    return out
