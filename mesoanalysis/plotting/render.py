import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from multiprocessing import Pool
from pathlib import Path

from .styles import STYLES


# ---------------------------------------------------------------------------
# NWS radar reflectivity colormap (14 colors, 5-dBZ bins from 5 to 75 dBZ)
# ---------------------------------------------------------------------------
_NWS_REFC_COLORS = [
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
_NWS_REFC_LEVELS = list(range(5, 80, 5))  # 5, 10, 15 ... 75


def _build_norm_and_cmap(style):
    """Return (norm, cmap) for a given style dict."""
    # --- NWS radar reflectivity ---
    if style.get("nws_radar"):
        cmap = mcolors.ListedColormap(_NWS_REFC_COLORS)
        norm = mcolors.BoundaryNorm(_NWS_REFC_LEVELS, cmap.N)
        return norm, cmap, _NWS_REFC_LEVELS

    # --- STP-style with explicit color list and BoundaryNorm ---
    if "cmap_colors" in style:
        cmap = mcolors.ListedColormap(style["cmap_colors"])
        levels = style["levels"]
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return norm, cmap, levels

    # --- Levels from explicit list ---
    if "levels" in style:
        levels = style["levels"]
        cmap = plt.get_cmap(style["cmap"])
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return norm, cmap, levels

    # --- Levels from range tuple ---
    if "levels_range" in style:
        lo, hi, step = style["levels_range"]
        levels = np.arange(lo, hi + step / 2, step)
        cmap = plt.get_cmap(style["cmap"]) if style.get("cmap") else None
        norm = None if cmap is None else mcolors.BoundaryNorm(levels, cmap.N)
        return norm, cmap, levels

    return None, None, None


def _render_one(args):
    """Render a single parameter map in a subprocess.

    matplotlib is NOT thread-safe, so this must run in a separate process
    via multiprocessing.Pool.  Each subprocess initialises its own Agg
    backend.
    """
    name, style, lon, lat, data, analysis_time, output_dir, model_name = args

    # Each subprocess needs its own backend
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mcolors

    # Re-import cartopy in subprocess
    import cartopy.crs as _ccrs
    import cartopy.feature as _cfeature

    # ---- build norm/cmap locally (same logic as _build_norm_and_cmap) ----
    norm, cmap, levels = None, None, None

    if style.get("nws_radar"):
        cmap = _mcolors.ListedColormap(_NWS_REFC_COLORS)
        norm = _mcolors.BoundaryNorm(_NWS_REFC_LEVELS, cmap.N)
        levels = _NWS_REFC_LEVELS
    elif "cmap_colors" in style:
        cmap = _mcolors.ListedColormap(style["cmap_colors"])
        levels = style["levels"]
        norm = _mcolors.BoundaryNorm(levels, cmap.N)
    elif "levels" in style:
        levels = style["levels"]
        cmap = _plt.get_cmap(style["cmap"])
        norm = _mcolors.BoundaryNorm(levels, cmap.N)
    elif "levels_range" in style:
        lo, hi, step = style["levels_range"]
        levels = np.arange(lo, hi + step / 2, step)
        cmap = _plt.get_cmap(style["cmap"]) if style.get("cmap") else None
        norm = None if cmap is None else _mcolors.BoundaryNorm(levels, cmap.N)

    # ---- create map ----
    proj = _ccrs.LambertConformal(central_longitude=-97.5)
    fig, ax = _plt.subplots(figsize=(12, 8), subplot_kw={"projection": proj})
    extent = [-125, -66, 24, 50]
    ax.set_extent(extent, crs=_ccrs.PlateCarree())
    ax.add_feature(_cfeature.STATES, linewidth=0.5)
    ax.add_feature(_cfeature.COASTLINE)
    ax.add_feature(_cfeature.BORDERS)

    transform = _ccrs.PlateCarree()
    contour_only = style.get("contour_only", False)
    extend = style.get("extend", "neither")

    if contour_only:
        cs = ax.contour(
            lon, lat, data,
            levels=levels,
            colors="k",
            linewidths=1.0,
            transform=transform,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    else:
        cf = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=extend,
            transform=transform,
        )
        _plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7,
                       label=style["label"])

    label = style["label"]
    time_str = analysis_time.strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(
        f"{model_name.upper()} + Sfc Obs Analysis | {label}\n{time_str}",
        fontsize=11,
    )

    output_path = Path(output_dir) / f"{name}.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    _plt.close(fig)
    return str(output_path)


def render_field(lon, lat, data, name, analysis_time, output_dir, model_name="HRRR"):
    """Render a single parameter field to PNG.

    Parameters
    ----------
    lon, lat : 2-D arrays
        Grid coordinates.
    data : 2-D array
        Parameter data on the grid.
    name : str
        Parameter name (key into STYLES).
    analysis_time : datetime
        Analysis valid time.
    output_dir : path-like
        Directory for output PNG.
    model_name : str
        Model display name for the title (e.g. ``"HRRR"``, ``"RAP"``, ``"NAM"``).
    """
    style = STYLES.get(name)
    if style is None:
        return

    norm, cmap, levels = _build_norm_and_cmap(style)

    from .maps import make_map
    fig, ax = make_map()
    transform = ccrs.PlateCarree()

    contour_only = style.get("contour_only", False)
    extend = style.get("extend", "neither")

    if contour_only:
        # Black contour lines only (e.g. MSLP)
        cs = ax.contour(
            lon, lat, data,
            levels=levels,
            colors="k",
            linewidths=1.0,
            transform=transform,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    else:
        cf = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=extend,
            transform=transform,
        )
        plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7,
                      label=style["label"])

    # Title
    label = style["label"]
    time_str = analysis_time.strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(
        f"{model_name.upper()} + Sfc Obs Analysis | {label}\n{time_str}",
        fontsize=11,
    )

    # Save
    output_path = Path(output_dir) / f"{name}.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def render_all(lon, lat, params_dict, analysis_time, output_dir, model_name="HRRR"):
    """Render all parameter fields that have a matching style definition.

    Uses multiprocessing.Pool to render maps in parallel across CPU cores.
    matplotlib is NOT thread-safe, so each render runs in a separate process.

    Parameters
    ----------
    lon, lat : 2-D arrays
        Grid coordinates.
    params_dict : dict
        Mapping of parameter name to 2-D data array.
    analysis_time : datetime
        Analysis valid time.
    output_dir : path-like
        Root output directory.
    model_name : str
        Model display name for titles (e.g. ``"HRRR"``, ``"RAP"``, ``"NAM"``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of (name, style, lon, lat, data, ...) tuples for pool.map
    render_args = []
    for name, data in params_dict.items():
        if name in STYLES:
            render_args.append((
                name, STYLES[name], lon, lat, data,
                analysis_time, str(output_dir), model_name,
            ))

    if not render_args:
        return

    n_workers = min(8, os.cpu_count() or 4, len(render_args))
    print(f"  rendering {len(render_args)} maps with {n_workers} workers ...")

    with Pool(processes=n_workers) as pool:
        for output_path in pool.imap_unordered(_render_one, render_args):
            print(f"  saved {output_path}")
