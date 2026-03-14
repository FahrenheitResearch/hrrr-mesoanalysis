"""
Multi-pass Barnes objective analysis.

Blends surface observations into HRRR background fields using a
successive-correction scheme with Gaussian distance weighting.

Uses metrust's Rust-native inverse_distance_to_points (kind=1, Barnes)
for the heavy lifting.

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

from mesoanalysis.config import DOMAIN

logger = logging.getLogger(__name__)


def _project_coords(
    lats: np.ndarray,
    lons: np.ndarray,
    proj: ccrs.Projection,
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic coordinates to *proj* x/y (metres)."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    original_shape = lats.shape

    pts = proj.transform_points(
        ccrs.PlateCarree(),
        lons.ravel(),
        lats.ravel(),
    )

    x = pts[:, 0].reshape(original_shape)
    y = pts[:, 1].reshape(original_shape)
    return x, y


def _build_grid_tree(grid_x, grid_y):
    """Build a KDTree from grid coordinates (once, reuse across passes)."""
    from scipy.spatial import cKDTree
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return cKDTree(pts)


def _interpolate_background_to_obs(tree, background_flat, obs_x, obs_y):
    """Nearest-neighbor interpolation of background to obs locations via KDTree."""
    obs_pts = np.column_stack([obs_x, obs_y])
    _, idx = tree.query(obs_pts)
    return background_flat[idx]


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
    using successive Gaussian-weighted correction passes. The core
    Barnes interpolation is done via metrust's Rust engine.

    Parameters
    ----------
    grid_lats, grid_lons : ndarray, shape (ny, nx)
    background : ndarray, shape (ny, nx)
    obs_lats, obs_lons : ndarray, shape (n_obs,)
    obs_values : ndarray, shape (n_obs,)
    kappa : float
        First-pass smoothing parameter (m^2).
    gamma : float
        Convergence factor; second-pass kappa is ``gamma * kappa``.
    passes : int
        Number of successive-correction passes (typically 2).
    projection : cartopy CRS, optional

    Returns
    -------
    analyzed : ndarray, shape (ny, nx)
    """
    from metrust._metrust import interpolate as _interp

    ny, nx = background.shape
    n_obs = len(obs_values)
    logger.info(
        "Barnes analysis: grid %dx%d, %d obs, kappa=%.3e, gamma=%.2f, passes=%d",
        ny, nx, n_obs, kappa, gamma, passes,
    )

    if n_obs == 0:
        logger.warning("No observations provided; returning background unchanged.")
        return background.copy()

    if projection is None:
        projection = DOMAIN.projection

    grid_x, grid_y = _project_coords(grid_lats, grid_lons, projection)
    obs_x, obs_y = _project_coords(obs_lats, obs_lons, projection)

    obs_x_flat = obs_x.ravel().astype(np.float64)
    obs_y_flat = obs_y.ravel().astype(np.float64)
    grid_x_flat = grid_x.ravel().astype(np.float64)
    grid_y_flat = grid_y.ravel().astype(np.float64)

    cutoff = 4.0 * np.sqrt(kappa)
    bg_flat = background.ravel().astype(np.float64)

    # Build KDTree once for grid-to-obs interpolation
    grid_tree = _build_grid_tree(grid_x, grid_y)

    # First pass: compute innovations and Barnes-interpolate to grid
    first_guess = _interpolate_background_to_obs(
        grid_tree, bg_flat, obs_x_flat, obs_y_flat,
    )
    innovations = (obs_values - first_guess).astype(np.float64)

    correction = np.asarray(_interp.inverse_distance_to_points(
        obs_x_flat, obs_y_flat, innovations,
        grid_x_flat, grid_y_flat,
        cutoff, 1, 1, kappa, gamma,
    ))
    correction = np.nan_to_num(correction, nan=0.0)
    analyzed_flat = bg_flat + correction

    # Subsequent passes
    current_kappa = kappa
    for p in range(2, passes + 1):
        current_kappa = gamma * current_kappa
        current_cutoff = 4.0 * np.sqrt(current_kappa)

        analysis_at_obs = _interpolate_background_to_obs(
            grid_tree, analyzed_flat, obs_x_flat, obs_y_flat,
        )
        innovations = (obs_values - analysis_at_obs).astype(np.float64)

        correction = np.asarray(_interp.inverse_distance_to_points(
            obs_x_flat, obs_y_flat, innovations,
            grid_x_flat, grid_y_flat,
            current_cutoff, 1, 1, current_kappa, gamma,
        ))
        correction = np.nan_to_num(correction, nan=0.0)
        analyzed_flat = analyzed_flat + correction

    return analyzed_flat.reshape(ny, nx)
