"""Export mesoanalysis grids as web-ready transparent PNG overlays.

Produces Leaflet-compatible image overlays: each parameter is rendered as a
transparent RGBA PNG on a regular lat/lon (EPSG:4326) grid, accompanied by a
manifest.json with bounding-box metadata and a compressed .npz archive of the
raw grids for point queries.

Target: full export of all 19 parameters in < 5 seconds.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.spatial import Delaunay

from mesoanalysis.config import DOMAIN
from mesoanalysis.plotting.styles import STYLES

# ---------------------------------------------------------------------------
# Target grid — regular lat/lon covering CONUS
# ---------------------------------------------------------------------------

_LON_MIN, _LON_MAX, _LAT_MIN, _LAT_MAX = DOMAIN.extent  # -125, -66, 24, 50

# ~0.03-deg resolution  ->  ~2000 x 870 pixels
_NX = 1967  # (125-66) / 0.03
_NY = 867   # (50-24) / 0.03

_TARGET_LONS = np.linspace(_LON_MIN, _LON_MAX, _NX)
_TARGET_LATS = np.linspace(_LAT_MIN, _LAT_MAX, _NY)
_TARGET_LON2D, _TARGET_LAT2D = np.meshgrid(_TARGET_LONS, _TARGET_LATS)

# Bounds in Leaflet convention: [[south, west], [north, east]]
_BOUNDS: List[List[float]] = [[_LAT_MIN, _LON_MIN], [_LAT_MAX, _LON_MAX]]


# ---------------------------------------------------------------------------
# NWS radar reflectivity colormap (duplicated from render.py to avoid
# importing matplotlib's full plotting stack at module level)
# ---------------------------------------------------------------------------

_NWS_REFC_COLORS_HEX = [
    "#04e9e7",  # 5
    "#019ff4",  # 10
    "#0300f4",  # 15
    "#02fd02",  # 20
    "#01c501",  # 25
    "#008e00",  # 30
    "#fdf802",  # 35
    "#e5bc00",  # 40
    "#fd9500",  # 45
    "#fd0000",  # 50
    "#d40000",  # 55
    "#bc0000",  # 60
    "#f800fd",  # 65
    "#9854c6",  # 70
]
_NWS_REFC_LEVELS = list(range(5, 80, 5))


# ---------------------------------------------------------------------------
# Color-scale helpers — pure NumPy / Pillow, no matplotlib
# ---------------------------------------------------------------------------

def _parse_hex(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _build_lut_from_mpl_cmap(cmap_name: str, n_colors: int = 256) -> NDArray:
    """Build a (256, 4) uint8 LUT from a matplotlib colormap name.

    We do a one-time import of matplotlib.colormaps here; the LUT is a plain
    numpy array used for all subsequent lookups — no matplotlib in the hot path.
    """
    import matplotlib as mpl
    cmap = mpl.colormaps[cmap_name]
    indices = np.linspace(0, 1, n_colors)
    rgba_float = cmap(indices)  # (n_colors, 4) float 0-1
    return (rgba_float * 255).astype(np.uint8)


def _levels_from_style(style: dict) -> Optional[NDArray]:
    """Extract contour levels from a style definition."""
    if style.get("nws_radar"):
        return np.asarray(_NWS_REFC_LEVELS, dtype=np.float64)
    if "levels" in style:
        return np.asarray(style["levels"], dtype=np.float64)
    if "levels_range" in style:
        lo, hi, step = style["levels_range"]
        return np.arange(lo, hi + step / 2, step)
    return None


# ---------------------------------------------------------------------------
# Colormap builders — return a function  data_array -> RGBA uint8 image
# ---------------------------------------------------------------------------

def _make_boundary_mapper(
    levels: NDArray,
    lut: NDArray,
    extend: str = "neither",
) -> callable:
    """Return a function that maps a float array to RGBA using BoundaryNorm logic.

    `lut` has shape (N_colors, 4) uint8.
    `levels` has len N_levels.  BoundaryNorm creates N_levels - 1 bins
    (plus optional under/over bins when extend != 'neither').

    Pixels below the lowest level are transparent (alpha=0) unless extend
    includes 'min'/'both'.  Similarly for above the highest level.
    """
    n_bins = len(levels) - 1  # interior bins
    has_under = extend in ("min", "both")
    has_over = extend in ("max", "both")

    # Total color slots needed
    n_total = n_bins + int(has_under) + int(has_over)

    # Resample LUT to n_total colors
    if lut.shape[0] != n_total:
        idx = np.linspace(0, lut.shape[0] - 1, n_total).astype(int)
        colors = lut[idx]  # (n_total, 4)
    else:
        colors = lut

    def mapper(data: NDArray) -> NDArray:
        """Map float array -> (H, W, 4) uint8 RGBA."""
        h, w = data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)  # transparent by default

        # Digitize: bin index for each value
        # np.digitize returns 0 for < levels[0], len(levels) for > levels[-1]
        bin_idx = np.digitize(data, levels)  # values 0 .. len(levels)

        valid = np.isfinite(data)

        if has_under:
            # Under bin gets color index 0; interior bins shift by 1
            color_offset = 1
            under_mask = valid & (bin_idx == 0)
            rgba[under_mask] = colors[0]
        else:
            color_offset = 0

        # Interior bins: digitize returns 1..n_bins for values in bin range
        for b in range(1, n_bins + 1):
            mask = valid & (bin_idx == b)
            rgba[mask] = colors[b - 1 + color_offset]

        if has_over:
            over_mask = valid & (bin_idx >= len(levels))
            rgba[over_mask] = colors[-1]

        return rgba

    return mapper


def _build_mapper(style: dict) -> Optional[callable]:
    """Build a data->RGBA mapper for a style definition.

    Returns None for contour-only styles (e.g. MSLP) which have no filled
    color rendering.
    """
    if style.get("contour_only"):
        return None

    levels = _levels_from_style(style)
    if levels is None:
        return None

    extend = style.get("extend", "neither")

    # Build the color LUT
    if style.get("nws_radar"):
        # Explicit color list — one color per bin
        rgb_list = [_parse_hex(h) for h in _NWS_REFC_COLORS_HEX]
        lut = np.array([(r, g, b, 255) for r, g, b in rgb_list], dtype=np.uint8)
    elif "cmap_colors" in style:
        rgb_list = [_parse_hex(h) for h in style["cmap_colors"]]
        lut = np.array([(r, g, b, 255) for r, g, b in rgb_list], dtype=np.uint8)
    else:
        cmap_name = style.get("cmap")
        if cmap_name is None:
            return None
        # 256-entry LUT from matplotlib colormap
        lut = _build_lut_from_mpl_cmap(cmap_name, n_colors=256)

    return _make_boundary_mapper(levels, lut, extend)


# ---------------------------------------------------------------------------
# Display metadata helpers
# ---------------------------------------------------------------------------

def _display_name(style: dict) -> str:
    """Extract a human display name from the style label."""
    label = style.get("label", "")
    # label is like "SBCAPE (J/kg)" — return the part before the units
    if "(" in label:
        return label[: label.index("(")].strip()
    return label


def _units(style: dict) -> str:
    """Extract units string from style label."""
    label = style.get("label", "")
    if "(" in label and ")" in label:
        return label[label.index("(") + 1 : label.index(")")]
    return ""


# ---------------------------------------------------------------------------
# Reprojection: Lambert Conformal curvilinear -> regular lat/lon
#
# Strategy: build the Delaunay triangulation of the source grid ONCE, then
# for each field use barycentric interpolation with precomputed weights.
# This turns O(N_fields * N_pts * log(N_pts)) into O(N_pts * log(N_pts))
# for triangulation + O(N_fields * N_target) for interpolation.
# ---------------------------------------------------------------------------

def _compute_interp_weights(
    src_lon: NDArray,
    src_lat: NDArray,
    subsample_step: int = 1,
) -> Tuple[Delaunay, NDArray, NDArray, NDArray]:
    """Precompute Delaunay triangulation and barycentric interpolation weights.

    Returns
    -------
    tri : Delaunay triangulation of (subsampled) source points
    vtx : (N_target, 3) int — indices of the 3 source vertices per target point
    wts : (N_target, 3) float — barycentric weights
    valid : (N_target,) bool — True where target lies inside the convex hull
    """
    # Normalize longitudes to -180..180 (some models use 0..360 convention)
    src_lon_norm = src_lon.ravel()[::subsample_step].copy()
    src_lon_norm[src_lon_norm > 180] -= 360

    src_points = np.column_stack((
        src_lon_norm,
        src_lat.ravel()[::subsample_step],
    ))

    tri = Delaunay(src_points)

    target_flat = np.column_stack((_TARGET_LON2D.ravel(), _TARGET_LAT2D.ravel()))
    n_target = target_flat.shape[0]

    # Find which simplex each target point falls in
    simplex_idx = tri.find_simplex(target_flat)
    valid = simplex_idx >= 0

    # For invalid points, use simplex 0 as placeholder (will be masked later)
    s_idx = np.where(valid, simplex_idx, 0)

    # Vertex indices (in subsampled source array) for each target point
    vtx = tri.simplices[s_idx]  # (n_target, 3)

    # Compute barycentric coordinates
    # For triangle with vertices A, B, C and point P:
    #   T = [[Bx-Ax, Cx-Ax], [By-Ay, Cy-Ay]]
    #   (lambda1, lambda2) = T^-1 @ (P - A)
    #   lambda0 = 1 - lambda1 - lambda2
    T = tri.transform[s_idx]  # (n_target, 2, 3) but actually (n_target, ndim+1, ndim)
    # tri.transform[i] has shape (ndim+1, ndim): first ndim rows are the
    # inverse transformation matrix, last row is the translation (vertex 0).
    delta = target_flat - T[:, 2, :]  # P - origin  (n_target, 2)
    bary = np.einsum("ijk,ik->ij", T[:, :2, :], delta)  # (n_target, 2)
    wts = np.column_stack((1 - bary.sum(axis=1), bary[:, 0], bary[:, 1]))

    return tri, vtx, wts, valid


def _apply_interp(
    data_flat: NDArray,
    vtx: NDArray,
    wts: NDArray,
    valid: NDArray,
    subsample_step: int,
) -> NDArray:
    """Apply precomputed barycentric weights to interpolate one field.

    Parameters
    ----------
    data_flat : 1-D source data (full resolution, will be subsampled)
    vtx, wts, valid : from _compute_interp_weights
    subsample_step : same step used when building triangulation

    Returns
    -------
    2-D array of shape (_NY, _NX) with NaN outside the convex hull.
    """
    src_vals = data_flat[::subsample_step]
    # Gather vertex values: (n_target, 3)
    v = src_vals[vtx]
    # Weighted sum
    result = np.sum(v * wts, axis=1)
    result[~valid] = np.nan
    return result.reshape(_NY, _NX)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_web_overlays(
    lon: NDArray,
    lat: NDArray,
    params: Dict[str, NDArray],
    analysis_time: datetime,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """Export all parameter fields as web-ready transparent PNG overlays.

    Parameters
    ----------
    lon, lat : 2-D arrays
        Source grid coordinates (curvilinear Lambert Conformal).
    params : dict
        Mapping of parameter name to 2-D data array on the source grid.
    analysis_time : datetime
        Analysis valid time.
    output_dir : path, optional
        Root output directory.  Defaults to ``output/{YYYYMMDD_HHMM}``.

    Returns
    -------
    Path to the output directory.
    """
    t0 = time.perf_counter()

    if output_dir is None:
        output_dir = Path(f"./output/{analysis_time:%Y%m%d_%H%M}")
    output_dir = Path(output_dir)

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    time_str = analysis_time.strftime("%Y%m%d_%H%M")

    # ------------------------------------------------------------------
    # Pre-build all color mappers (one-time matplotlib import for LUTs)
    # ------------------------------------------------------------------
    mappers: Dict[str, callable] = {}
    for name in params:
        style = STYLES.get(name)
        if style is None:
            continue
        mapper = _build_mapper(style)
        if mapper is not None:
            mappers[name] = mapper

    # ------------------------------------------------------------------
    # Reproject all parameter grids to regular lat/lon
    # ------------------------------------------------------------------
    print(f"  Reprojecting {len(mappers)} fields to EPSG:4326 ...")
    t_reproj = time.perf_counter()

    # Subsample source grid for triangulation speed.  Model grids can have ~1.8M
    # points; subsampling by 7 gives ~37k points — plenty for smooth interp.
    n_src = lon.size
    subsample_step = max(1, n_src // 250_000)

    # Build triangulation once — or load cached weights if grid hasn't changed.
    # The cache key is a hash of (subsampled lon/lat + target grid dims).
    cache_dir = Path("./output/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    grid_hash = hashlib.sha256(
        lon.ravel()[::subsample_step].tobytes()
        + lat.ravel()[::subsample_step].tobytes()
        + np.array([_NY, _NX], dtype=np.int64).tobytes()
    ).hexdigest()[:16]
    cache_path = cache_dir / f"interp_weights_{grid_hash}.npz"

    if cache_path.exists():
        print("  Loading cached interpolation weights ...")
        cached = np.load(str(cache_path))
        vtx = cached["vtx"]
        wts = cached["wts"]
        valid = cached["valid"]
    else:
        _, vtx, wts, valid = _compute_interp_weights(lon, lat, subsample_step)
        np.savez(
            str(cache_path),
            vtx=vtx, wts=wts, valid=valid,
        )
        print(f"  Cached interpolation weights to {cache_path}")

    # Apply to each field
    reprojected: Dict[str, NDArray] = {}
    for name in mappers:
        data_flat = params[name].ravel().astype(np.float64)
        reprojected[name] = _apply_interp(data_flat, vtx, wts, valid, subsample_step)

    dt_reproj = time.perf_counter() - t_reproj
    print(f"  Reprojection done in {dt_reproj:.1f}s")

    # ------------------------------------------------------------------
    # Render transparent PNGs
    # ------------------------------------------------------------------
    print(f"  Rendering {len(mappers)} transparent PNGs ...")
    t_render = time.perf_counter()

    manifest_params: Dict[str, Dict[str, Any]] = {}

    for name, mapper in mappers.items():
        data = reprojected[name]
        style = STYLES[name]

        # Map data to RGBA
        # Image convention: row 0 = top = north, so flip vertically
        rgba = mapper(np.flipud(data))

        # Save as PNG via Pillow (much faster than matplotlib for raw RGBA)
        img = Image.fromarray(rgba, mode="RGBA")
        png_path = tiles_dir / f"{name}.png"
        img.save(str(png_path), compress_level=1)

        # Compute stats on the original (non-NaN) data for manifest
        finite = data[np.isfinite(data)]
        data_min = float(finite.min()) if finite.size > 0 else None
        data_max = float(finite.max()) if finite.size > 0 else None

        manifest_params[name] = {
            "display_name": _display_name(style),
            "units": _units(style),
            "min": round(data_min, 2) if data_min is not None else None,
            "max": round(data_max, 2) if data_max is not None else None,
            "png_path": f"tiles/{name}.png",
        }

    dt_render = time.perf_counter() - t_render
    print(f"  PNG rendering done in {dt_render:.1f}s")

    # ------------------------------------------------------------------
    # Save manifest.json
    # ------------------------------------------------------------------
    manifest = {
        "analysis_time": analysis_time.isoformat(),
        "bounds": _BOUNDS,
        "grid_shape": [_NY, _NX],
        "parameters": manifest_params,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote {manifest_path}")

    # ------------------------------------------------------------------
    # Save compressed grids for point queries
    # ------------------------------------------------------------------
    grids_path = output_dir / "grids.npz"
    grid_arrays = {
        "_lons": _TARGET_LONS,
        "_lats": _TARGET_LATS,
    }
    for name, data in reprojected.items():
        grid_arrays[name] = data.astype(np.float32)
    np.savez(str(grids_path), **grid_arrays)
    print(f"  Wrote {grids_path}")

    dt_total = time.perf_counter() - t0
    print(f"  Export complete: {len(mappers)} overlays in {dt_total:.1f}s")

    return output_dir
