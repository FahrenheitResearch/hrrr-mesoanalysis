import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def make_map(figsize=(12, 8), extent=None):
    """Create a figure with Lambert Conformal projection and standard features."""
    proj = ccrs.LambertConformal(central_longitude=-97.5)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    if extent is None:
        extent = [-125, -66, 24, 50]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    return fig, ax
