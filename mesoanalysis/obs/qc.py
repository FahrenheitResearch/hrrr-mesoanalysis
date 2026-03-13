"""
Quality control for surface observations.

Two passes:
1. **Range checks** -- reject obs with values outside physically reasonable bounds.
2. **Gross-error (buddy) check** against a HRRR first-guess field -- reject obs
   that deviate from the model background by more than a threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from mesoanalysis.obs.models import ObsCollection

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeLimits:
    """Physically reasonable bounds for each field."""

    t_min: float = -60.0    # C
    t_max: float = 55.0     # C
    td_min: float = -70.0   # C
    td_max: float = 40.0    # C
    wspd_max: float = 60.0  # m/s
    mslp_min: float = 940.0  # hPa
    mslp_max: float = 1060.0  # hPa


@dataclass(frozen=True)
class GrossErrorThresholds:
    """Maximum allowed |obs - model| per field."""

    t_C: float = 15.0       # C
    td_C: float = 20.0      # C
    wind_ms: float = 15.0   # m/s  (applied to both u and v)
    mslp_hPa: float = 10.0  # hPa


# ---------------------------------------------------------------------------
# Range check
# ---------------------------------------------------------------------------

def _range_check(
    obs: ObsCollection,
    limits: RangeLimits = RangeLimits(),
) -> np.ndarray:
    """Return a boolean mask (True = pass) for range checks."""
    t = obs.t_array
    td = obs.td_array
    wspd = obs.wind_speed_array
    mslp = obs.mslp_array

    good = (
        (t >= limits.t_min) & (t <= limits.t_max)
        & (td >= limits.td_min) & (td <= limits.td_max)
        & (td <= t)
        & (wspd >= 0) & (wspd <= limits.wspd_max)
        & (mslp >= limits.mslp_min) & (mslp <= limits.mslp_max)
    )
    return good


# ---------------------------------------------------------------------------
# Gross-error (buddy) check against HRRR first guess
# ---------------------------------------------------------------------------

@dataclass
class HRRRFirstGuess:
    """Container for 2-D HRRR surface fields used as a background check.

    All arrays are 2-D with shape (ny, nx).  ``lats_1d`` and ``lons_1d``
    must be monotonically increasing 1-D coordinate vectors suitable for
    `scipy.interpolate.RegularGridInterpolator`.

    If the HRRR grid has *decreasing* latitudes (north-to-south), the caller
    must flip them (and the corresponding data axis) before passing here.
    """

    lats_1d: np.ndarray   # (ny,) -- must be monotonically increasing
    lons_1d: np.ndarray   # (nx,) -- must be monotonically increasing
    t_2m: np.ndarray      # 2-m temperature, C    (ny, nx)
    td_2m: np.ndarray     # 2-m dewpoint, C       (ny, nx)
    u_10m: np.ndarray     # 10-m u-wind, m/s      (ny, nx)
    v_10m: np.ndarray     # 10-m v-wind, m/s      (ny, nx)
    mslp: np.ndarray      # mean sea-level pressure, hPa  (ny, nx)


def _interp_field(
    field: np.ndarray,
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    obs_lats: np.ndarray,
    obs_lons: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of a 2-D field to observation locations."""
    interp = RegularGridInterpolator(
        (lats_1d, lons_1d),
        field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([obs_lats, obs_lons])
    return interp(pts)


def _gross_error_check(
    obs: ObsCollection,
    hrrr: HRRRFirstGuess,
    thresholds: GrossErrorThresholds = GrossErrorThresholds(),
) -> np.ndarray:
    """Return boolean mask (True = pass) for gross-error check vs HRRR."""
    lats = obs.lats
    lons = obs.lons

    t_bg = _interp_field(hrrr.t_2m, hrrr.lats_1d, hrrr.lons_1d, lats, lons)
    td_bg = _interp_field(hrrr.td_2m, hrrr.lats_1d, hrrr.lons_1d, lats, lons)
    u_bg = _interp_field(hrrr.u_10m, hrrr.lats_1d, hrrr.lons_1d, lats, lons)
    v_bg = _interp_field(hrrr.v_10m, hrrr.lats_1d, hrrr.lons_1d, lats, lons)
    mslp_bg = _interp_field(hrrr.mslp, hrrr.lats_1d, hrrr.lons_1d, lats, lons)

    t_ok = np.abs(obs.t_array - t_bg) <= thresholds.t_C
    td_ok = np.abs(obs.td_array - td_bg) <= thresholds.td_C
    u_ok = np.abs(obs.u_array - u_bg) <= thresholds.wind_ms
    v_ok = np.abs(obs.v_array - v_bg) <= thresholds.wind_ms
    mslp_ok = np.abs(obs.mslp_array - mslp_bg) <= thresholds.mslp_hPa

    # Where background is NaN (station outside HRRR domain), pass by default
    nan_mask = np.isnan(t_bg) | np.isnan(td_bg) | np.isnan(mslp_bg)
    good = t_ok & td_ok & u_ok & v_ok & mslp_ok
    good[nan_mask] = True

    return good


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def qc_obs(
    obs: ObsCollection,
    hrrr: Optional[HRRRFirstGuess] = None,
    range_limits: RangeLimits = RangeLimits(),
    ge_thresholds: GrossErrorThresholds = GrossErrorThresholds(),
    verbose: bool = True,
) -> ObsCollection:
    """Run quality control on an ObsCollection.

    Parameters
    ----------
    obs : ObsCollection
        Raw observations.
    hrrr : HRRRFirstGuess | None
        HRRR first-guess fields.  If ``None``, the gross-error check is skipped.
    range_limits : RangeLimits
        Bounds for the range check.
    ge_thresholds : GrossErrorThresholds
        Thresholds for the gross-error check.
    verbose : bool
        If True, print QC summary statistics.

    Returns
    -------
    ObsCollection
        Filtered observations that passed all QC steps.
    """
    n_total = len(obs)
    if n_total == 0:
        log.info("QC: no observations to check.")
        return obs

    # --- Pass 1: range check ------------------------------------------------
    range_mask = _range_check(obs, range_limits)
    n_range_fail = int(np.sum(~range_mask))

    # --- Pass 2: gross-error check ------------------------------------------
    if hrrr is not None:
        ge_mask = _gross_error_check(obs, hrrr, ge_thresholds)
        n_ge_fail = int(np.sum(range_mask & ~ge_mask))  # only among range-pass
    else:
        ge_mask = np.ones(n_total, dtype=bool)
        n_ge_fail = 0

    combined = range_mask & ge_mask
    n_pass = int(np.sum(combined))

    # --- Summary ------------------------------------------------------------
    summary = (
        f"QC summary: {n_total} obs in  |  "
        f"range-fail {n_range_fail}  |  "
        f"gross-error-fail {n_ge_fail}  |  "
        f"{n_pass} pass ({100 * n_pass / n_total:.1f}%)"
    )
    log.info(summary)
    if verbose:
        print(summary)

    return obs.filter(combined)
