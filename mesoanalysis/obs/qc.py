"""
Quality control for surface observations.

Three passes:
1. **Range checks** -- reject obs with values outside physically reasonable bounds.
2. **Gross-error check** against model first-guess -- reject obs that deviate from
   the model background by more than a threshold.  Uses KDTree nearest-neighbor
   interpolation on the native curvilinear model grid (no 1-D approximation).
3. **Spatial consistency (buddy) check** -- reject isolated outliers where the obs
   departs from both the model AND from nearby stations.

References
----------
SPC Mesoanalysis QC thresholds adapted from:
    Bothwell, P. D., J. A. Hart, and R. L. Thompson, 2002: An integrated
    three-dimensional objective analysis scheme in use at the Storm
    Prediction Center. 21st Conf. on Severe Local Storms, San Antonio, TX.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from mesoanalysis.config import QC_THRESHOLDS
from mesoanalysis.obs.models import ObsCollection

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds -- tuned to SPC mesoanalysis standards
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeLimits:
    """Physically reasonable bounds for each field."""

    t_min: float = -50.0    # C
    t_max: float = 55.0     # C
    td_min: float = -60.0   # C
    td_max: float = 35.0    # C
    wspd_max: float = 50.0  # m/s  (~97 kt -- rejects erroneous 999-kt reports)
    mslp_min: float = 945.0  # hPa
    mslp_max: float = 1055.0  # hPa
    td_depression_max: float = 50.0  # C  (T - Td must be < this)


@dataclass(frozen=True)
class GrossErrorThresholds:
    """Maximum allowed |obs - model| per field.

    Tightened from the original (15/20/15/10) to SPC-grade values.
    These reject true outliers while allowing real mesoscale departures.
    """

    t_C: float = 8.0        # C  (~14 F -- allows real cold-pool / warm-sector departures)
    td_C: float = 10.0      # C  (~18 F -- allows dryline / moisture-surge departures)
    wind_ms: float = 12.0   # m/s (~23 kt, applied to both u and v)
    mslp_hPa: float = 6.0   # hPa (allows mesoscale pressure features)


# ---------------------------------------------------------------------------
# Model first-guess container
# ---------------------------------------------------------------------------

@dataclass
class ModelFirstGuess:
    """Model surface fields for QC background comparison.

    Accepts the native curvilinear 2-D coordinate arrays -- no 1-D
    approximation needed.  Internally builds a KDTree for interpolation.
    """

    lats_2d: np.ndarray   # (ny, nx) -- native model latitudes
    lons_2d: np.ndarray   # (ny, nx) -- native model longitudes (-180..180)
    t_2m: np.ndarray      # 2-m temperature, C    (ny, nx)
    td_2m: np.ndarray     # 2-m dewpoint, C       (ny, nx)
    u_10m: np.ndarray     # 10-m u-wind, m/s      (ny, nx)
    v_10m: np.ndarray     # 10-m v-wind, m/s      (ny, nx)
    mslp: np.ndarray      # mean sea-level pressure, hPa  (ny, nx)

    _tree: Optional[cKDTree] = None

    def _ensure_tree(self):
        if self._tree is None:
            pts = np.column_stack([
                self.lats_2d.ravel(),
                self.lons_2d.ravel(),
            ])
            self._tree = cKDTree(pts)

    def interp_to_obs(self, obs_lats: np.ndarray, obs_lons: np.ndarray) -> dict:
        """Nearest-neighbor interpolation of all fields to obs locations.

        Returns dict with keys 't', 'td', 'u', 'v', 'mslp', each a 1-D array.
        """
        self._ensure_tree()
        pts = np.column_stack([obs_lats, obs_lons])
        _, idx = self._tree.query(pts)

        return {
            "t": self.t_2m.ravel()[idx],
            "td": self.td_2m.ravel()[idx],
            "u": self.u_10m.ravel()[idx],
            "v": self.v_10m.ravel()[idx],
            "mslp": self.mslp.ravel()[idx],
        }


# Backward-compatible alias
HRRRFirstGuess = ModelFirstGuess


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

    # NaN mslp (missing pressure) passes the range check.
    mslp_ok = np.isnan(mslp) | ((mslp >= limits.mslp_min) & (mslp <= limits.mslp_max))

    good = (
        (t >= limits.t_min) & (t <= limits.t_max)
        & (td >= limits.td_min) & (td <= limits.td_max)
        & (td <= t)
        & ((t - td) <= limits.td_depression_max)
        & (wspd >= 0) & (wspd <= limits.wspd_max)
        & mslp_ok
    )
    return good


# ---------------------------------------------------------------------------
# Gross-error check -- proper KDTree interpolation
# ---------------------------------------------------------------------------

def _gross_error_check(
    obs: ObsCollection,
    first_guess: ModelFirstGuess,
    thresholds: GrossErrorThresholds = GrossErrorThresholds(),
) -> np.ndarray:
    """Return boolean mask (True = pass) for gross-error check vs model."""
    bg = first_guess.interp_to_obs(obs.lats, obs.lons)

    t_ok = np.abs(obs.t_array - bg["t"]) <= thresholds.t_C
    td_ok = np.abs(obs.td_array - bg["td"]) <= thresholds.td_C
    u_ok = np.abs(obs.u_array - bg["u"]) <= thresholds.wind_ms
    v_ok = np.abs(obs.v_array - bg["v"]) <= thresholds.wind_ms

    # Pressure: NaN obs (missing) pass; valid obs must be within threshold
    obs_mslp = obs.mslp_array
    mslp_ok = np.isnan(obs_mslp) | (np.abs(obs_mslp - bg["mslp"]) <= thresholds.mslp_hPa)

    good = t_ok & td_ok & u_ok & v_ok & mslp_ok
    return good


# ---------------------------------------------------------------------------
# Spatial consistency (buddy) check
# ---------------------------------------------------------------------------

def _buddy_check(
    obs: ObsCollection,
    max_buddy_dist_deg: float = 1.5,
    min_buddies: int = 3,
    t_spread: float = 6.0,     # C -- max allowed deviation from buddy median
    td_spread: float = 8.0,    # C
) -> np.ndarray:
    """Reject obs that disagree with nearby stations.

    For each station, find neighbors within max_buddy_dist_deg.  If >=
    min_buddies exist and the station departs from the neighbor median by
    more than the spread threshold, reject it.
    """
    n = len(obs)
    if n < min_buddies + 1:
        return np.ones(n, dtype=bool)

    pts = np.column_stack([obs.lats, obs.lons])
    tree = cKDTree(pts)

    t = obs.t_array
    td = obs.td_array
    good = np.ones(n, dtype=bool)

    for i in range(n):
        neighbors = tree.query_ball_point(pts[i], max_buddy_dist_deg)
        neighbors = [j for j in neighbors if j != i]

        if len(neighbors) < min_buddies:
            continue  # not enough buddies to judge -- keep

        t_med = np.median(t[neighbors])
        td_med = np.median(td[neighbors])

        if abs(t[i] - t_med) > t_spread:
            good[i] = False
        elif abs(td[i] - td_med) > td_spread:
            good[i] = False

    return good


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def qc_obs(
    obs: ObsCollection,
    first_guess: Optional[ModelFirstGuess] = None,
    range_limits: RangeLimits = RangeLimits(),
    ge_thresholds: GrossErrorThresholds = GrossErrorThresholds(),
    model: str = "hrrr",
    verbose: bool = True,
    # Legacy parameter name kept for backward compat
    hrrr: Optional[ModelFirstGuess] = None,
) -> ObsCollection:
    """Run quality control on an ObsCollection.

    Three-pass QC:
    1. Range check -- physical bounds
    2. Gross-error check -- departure from model background
    3. Buddy check -- spatial consistency with nearby stations

    Parameters
    ----------
    obs : ObsCollection
        Raw observations.
    first_guess : ModelFirstGuess, optional
        Model background for gross-error check.
    range_limits : RangeLimits
        Physical range bounds.
    ge_thresholds : GrossErrorThresholds
        Gross-error thresholds.  If not customised, auto-selected from
        ``QC_THRESHOLDS`` based on *model*.
    model : str
        Model identifier (``"hrrr"``, ``"rap"``, ``"nam"``).  Used to
        select default gross-error thresholds when *ge_thresholds* is
        at its default value.
    verbose : bool
        Print summary to stdout.
    hrrr : ModelFirstGuess, optional
        **Deprecated** alias for *first_guess* (backward compat).
    """
    # Handle legacy 'hrrr' parameter name
    if first_guess is None and hrrr is not None:
        first_guess = hrrr

    # Auto-select thresholds from config if using defaults
    if ge_thresholds == GrossErrorThresholds():
        model_lower = model.lower()
        if model_lower in QC_THRESHOLDS:
            qc_cfg = QC_THRESHOLDS[model_lower]
            ge_thresholds = GrossErrorThresholds(
                t_C=qc_cfg.t_C,
                td_C=qc_cfg.td_C,
                wind_ms=qc_cfg.wind_ms,
                mslp_hPa=qc_cfg.mslp_hPa,
            )

    n_total = len(obs)
    if n_total == 0:
        log.info("QC: no observations to check.")
        return obs

    # --- Pass 1: range check ------------------------------------------------
    range_mask = _range_check(obs, range_limits)
    n_range_fail = int(np.sum(~range_mask))

    # --- Pass 2: gross-error check ------------------------------------------
    if first_guess is not None:
        ge_mask = _gross_error_check(obs, first_guess, ge_thresholds)
        n_ge_fail = int(np.sum(range_mask & ~ge_mask))
    else:
        ge_mask = np.ones(n_total, dtype=bool)
        n_ge_fail = 0

    combined = range_mask & ge_mask

    # --- Pass 3: buddy check on survivors -----------------------------------
    survivors = obs.filter(combined)
    if len(survivors) > 10:
        buddy_mask = _buddy_check(survivors)
        n_buddy_fail = int(np.sum(~buddy_mask))
        survivors = survivors.filter(buddy_mask)
    else:
        n_buddy_fail = 0

    n_pass = len(survivors)

    # --- Summary ------------------------------------------------------------
    summary = (
        f"QC summary: {n_total} obs in  |  "
        f"range-fail {n_range_fail}  |  "
        f"gross-error-fail {n_ge_fail}  |  "
        f"buddy-fail {n_buddy_fail}  |  "
        f"{n_pass} pass ({100 * n_pass / n_total:.1f}%)"
    )
    log.info(summary)
    if verbose:
        print(summary)

    return survivors
