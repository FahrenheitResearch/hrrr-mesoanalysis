"""
Multi-pass Barnes objective analysis.

Blends surface observations into HRRR background fields using a
successive-correction scheme with Gaussian distance weighting.

References
----------
Barnes, S. L., 1964: A technique for maximizing details in numerical
    weather map analysis. J. Appl. Meteor., 3, 396-409.
"""

from __future__ import annotations

import logging
from typing import Optional

import cartopy.crs as ccrs
import numpy as np
from scipy.spatial import cKDTree

from mesoanalysis.config import DOMAIN

logger = logging.getLogger(__name__)

# Grid-point chunk size for batched KDTree queries (limits peak memory).
_CHUNK_SIZE = 200_000


def _project_coords(
    lats: np.ndarray,
    lons: np.ndarray,
    proj: ccrs.Projection,
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic coordinates to *proj* x/y (metres).

    Parameters
    ----------
    lats, lons : array-like
        Latitude / longitude arrays (any shape).
    proj : cartopy CRS
        Target projected coordinate system.

    Returns
    -------
    x, y : np.ndarray
        Projected coordinates (same shape as input).
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    original_shape = lats.shape

    pts = proj.transform_points(
        ccrs.PlateCarree(),
        lons.ravel(),
        lats.ravel(),
    )  # shape (N, 3) — columns are x, y, z

    x = pts[:, 0].reshape(original_shape)
    y = pts[:, 1].reshape(original_shape)
    return x, y


def _interpolate_background_to_obs(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    background: np.ndarray,
    obs_x: np.ndarray,
    obs_y: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate *background* to observation locations.

    Uses scipy's RegularGridInterpolator when the grid is regular, but
    for curvilinear HRRR grids we fall back to nearest-neighbour via a
    KDTree lookup (much faster than true bilinear on an unstructured
    grid and sufficient for innovation computation).
    """
    from scipy.interpolate import griddata

    ny, nx = background.shape
    pts_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    vals = background.ravel()

    # Use linear griddata; fall back to nearest for any NaNs.
    result = griddata(pts_grid, vals, (obs_x, obs_y), method="linear")

    # Fill any NaN (extrapolation) with nearest-neighbour.
    nans = np.isnan(result)
    if nans.any():
        nearest = griddata(pts_grid, vals, (obs_x[nans], obs_y[nans]), method="nearest")
        result[nans] = nearest

    return result


def _single_pass(
    grid_xy: np.ndarray,
    tree: cKDTree,
    innovations: np.ndarray,
    background: np.ndarray,
    kappa: float,
    cutoff: float,
) -> np.ndarray:
    """Apply one Barnes correction pass over the entire grid.

    Parameters
    ----------
    grid_xy : ndarray, shape (M, 2)
        Flattened grid point positions in projected metres.
    tree : cKDTree
        KDTree built from station projected positions.
    innovations : ndarray, shape (n_obs,)
        obs - first_guess differences.
    background : ndarray, shape (M,)
        Flattened field to correct.
    kappa : float
        Smoothing parameter for this pass (m^2).
    cutoff : float
        Search radius (metres).

    Returns
    -------
    analyzed : ndarray, shape (M,)
        Corrected field.
    """
    n_grid = grid_xy.shape[0]
    correction = np.zeros(n_grid, dtype=np.float64)
    four_kappa = 4.0 * kappa

    for start in range(0, n_grid, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, n_grid)
        chunk_xy = grid_xy[start:end]

        # For each grid point in the chunk, find stations within cutoff.
        neighbours = tree.query_ball_point(chunk_xy, r=cutoff)

        for j, nb_idx in enumerate(neighbours):
            if len(nb_idx) == 0:
                continue
            nb_idx = np.asarray(nb_idx)
            # Distance from this grid point to each neighbour station.
            dx = chunk_xy[j, 0] - tree.data[nb_idx, 0]
            dy = chunk_xy[j, 1] - tree.data[nb_idx, 1]
            r2 = dx * dx + dy * dy
            weights = np.exp(-r2 / four_kappa)
            w_sum = weights.sum()
            if w_sum > 0.0:
                correction[start + j] = np.dot(weights, innovations[nb_idx]) / w_sum

    return background + correction


def barnes_analysis(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    background: np.ndarray,
    obs_lats: np.ndarray,
    obs_lons: np.ndarray,
    obs_values: np.ndarray,
    kappa: float = 5.052e8,
    gamma: float = 0.3,
    passes: int = 2,
    projection: Optional[ccrs.Projection] = None,
) -> np.ndarray:
    """Multi-pass Barnes objective analysis.

    Blends point observations into a gridded background (e.g. HRRR)
    using successive Gaussian-weighted correction passes.

    Parameters
    ----------
    grid_lats : ndarray, shape (ny, nx)
        Grid latitude values (degrees N).
    grid_lons : ndarray, shape (ny, nx)
        Grid longitude values (degrees E).
    background : ndarray, shape (ny, nx)
        Background (first-guess) gridded field.
    obs_lats : ndarray, shape (n_obs,)
        Observation latitudes.
    obs_lons : ndarray, shape (n_obs,)
        Observation longitudes.
    obs_values : ndarray, shape (n_obs,)
        Observed values (same units / variable as *background*).
    kappa : float
        First-pass smoothing parameter (m^2).  Default tuned for ~90 km
        average station spacing.
    gamma : float
        Convergence factor; second-pass kappa is ``gamma * kappa``.
    passes : int
        Number of successive-correction passes (typically 2).
    projection : cartopy CRS, optional
        Map projection for distance calculations.  Defaults to the
        domain LambertConformal from ``config.DOMAIN``.

    Returns
    -------
    analyzed : ndarray, shape (ny, nx)
        The analyzed field.
    """
    ny, nx = background.shape
    n_obs = len(obs_values)
    logger.info(
        "Barnes analysis: grid %dx%d, %d obs, kappa=%.3e, gamma=%.2f, passes=%d",
        ny, nx, n_obs, kappa, gamma, passes,
    )

    if n_obs == 0:
        logger.warning("No observations provided; returning background unchanged.")
        return background.copy()

    # -- projection setup ----------------------------------------------------
    if projection is None:
        projection = DOMAIN.projection

    grid_x, grid_y = _project_coords(grid_lats, grid_lons, projection)
    obs_x, obs_y = _project_coords(obs_lats, obs_lons, projection)

    # -- build KDTree from station positions ---------------------------------
    station_xy = np.column_stack([obs_x.ravel(), obs_y.ravel()])
    tree = cKDTree(station_xy)

    # -- flatten grid for vectorised processing ------------------------------
    grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    bg_flat = background.ravel().astype(np.float64)

    # -- first pass ----------------------------------------------------------
    first_guess = _interpolate_background_to_obs(
        grid_x, grid_y, background, obs_x, obs_y,
    )
    innovations = obs_values - first_guess
    logger.debug(
        "Pass 1 innovations: mean=%.3f, std=%.3f", innovations.mean(), innovations.std(),
    )

    cutoff = 4.0 * np.sqrt(kappa)
    analyzed_flat = _single_pass(grid_xy, tree, innovations, bg_flat, kappa, cutoff)

    # -- subsequent passes ---------------------------------------------------
    current_kappa = kappa
    for p in range(2, passes + 1):
        current_kappa = gamma * current_kappa
        current_cutoff = 4.0 * np.sqrt(current_kappa)

        # Recompute innovations from latest analysis.
        analysis_at_obs = _interpolate_background_to_obs(
            grid_x, grid_y, analyzed_flat.reshape(ny, nx), obs_x, obs_y,
        )
        innovations = obs_values - analysis_at_obs
        logger.debug(
            "Pass %d innovations: mean=%.3f, std=%.3f",
            p, innovations.mean(), innovations.std(),
        )

        analyzed_flat = _single_pass(
            grid_xy, tree, innovations, analyzed_flat, current_kappa, current_cutoff,
        )

    return analyzed_flat.reshape(ny, nx)
