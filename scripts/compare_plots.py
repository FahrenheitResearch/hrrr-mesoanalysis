"""Generate HRRR vs Mesoanalysis comparison plots."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timezone, timedelta

now = datetime.now(tz=timezone.utc)
analysis_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)

from mesoanalysis.hrrr.ingest import load_hrrr
hrrr = load_hrrr(analysis_time, fxx=0)

raw_t2m_f = (hrrr.t2m_K - 273.15) * 9/5 + 32
raw_td2m_f = (hrrr.td2m_K - 273.15) * 9/5 + 32

from mesoanalysis.obs.fetch_multi import fetch_all_surface_obs
from mesoanalysis.obs.qc import qc_obs, HRRRFirstGuess
from mesoanalysis.config import BARNES
from mesoanalysis.analysis.barnes import barnes_analysis

obs = fetch_all_surface_obs(analysis_time, window_minutes=20)

t2m_C = hrrr.t2m_K - 273.15
td2m_C = hrrr.td2m_K - 273.15
mslp_hPa = hrrr.mslp_Pa / 100.0

# Normalize HRRR lons to -180..180
lons_norm = hrrr.lon.copy()
lons_norm[lons_norm > 180] -= 360

first_guess = HRRRFirstGuess(
    lats_2d=hrrr.lat,
    lons_2d=lons_norm,
    t_2m=t2m_C,
    td_2m=td2m_C,
    u_10m=hrrr.u10,
    v_10m=hrrr.v10,
    mslp=mslp_hPa,
)
obs_qc = qc_obs(obs, first_guess)
print(f"{len(obs_qc)} obs after QC")

analyzed_t2m = barnes_analysis(
    hrrr.lat, hrrr.lon, t2m_C,
    obs_qc.lats, obs_qc.lons, obs_qc.t_array,
    kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes,
)
analyzed_td2m = barnes_analysis(
    hrrr.lat, hrrr.lon, td2m_C,
    obs_qc.lats, obs_qc.lons, obs_qc.td_array,
    kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes,
)

meso_t2m_f = analyzed_t2m * 9/5 + 32
meso_td2m_f = analyzed_td2m * 9/5 + 32
diff_t = meso_t2m_f - raw_t2m_f
diff_td = meso_td2m_f - raw_td2m_f

lon_plot = hrrr.lon.copy()
lon_plot[lon_plot > 180] -= 360
proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)
t_levels = np.arange(-15, 111, 5)
diff_levels = np.arange(-8, 8.5, 1)


def make_4panel(extent_box, title_prefix, filename, obs_size=2):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={"projection": proj})

    for ax in axes.flat:
        ax.set_extent(extent_box, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, edgecolor="#666", linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.6)

    cf = axes[0, 0].contourf(
        lon_plot, hrrr.lat, raw_t2m_f,
        levels=t_levels, cmap="RdYlBu_r",
        transform=ccrs.PlateCarree(), extend="both",
    )
    plt.colorbar(cf, ax=axes[0, 0], orientation="horizontal", pad=0.03, aspect=30, shrink=0.85, label="deg F")
    axes[0, 0].set_title("Raw HRRR  |  2m Temperature", fontsize=14, fontweight="bold")

    cf = axes[0, 1].contourf(
        lon_plot, hrrr.lat, meso_t2m_f,
        levels=t_levels, cmap="RdYlBu_r",
        transform=ccrs.PlateCarree(), extend="both",
    )
    plt.colorbar(cf, ax=axes[0, 1], orientation="horizontal", pad=0.03, aspect=30, shrink=0.85, label="deg F")
    axes[0, 1].set_title("Mesoanalysis  |  2m Temperature", fontsize=14, fontweight="bold")

    cf = axes[1, 0].contourf(
        lon_plot, hrrr.lat, diff_t,
        levels=diff_levels, cmap="bwr",
        transform=ccrs.PlateCarree(), extend="both",
    )
    axes[1, 0].scatter(
        [o.lon for o in obs_qc], [o.lat for o in obs_qc],
        c="black", s=obs_size, alpha=0.4,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    plt.colorbar(cf, ax=axes[1, 0], orientation="horizontal", pad=0.03, aspect=30, shrink=0.85, label="deg F")
    axes[1, 0].set_title("Temperature Correction  (Meso - HRRR)", fontsize=14, fontweight="bold")

    cf = axes[1, 1].contourf(
        lon_plot, hrrr.lat, diff_td,
        levels=diff_levels, cmap="bwr",
        transform=ccrs.PlateCarree(), extend="both",
    )
    axes[1, 1].scatter(
        [o.lon for o in obs_qc], [o.lat for o in obs_qc],
        c="black", s=obs_size, alpha=0.4,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    plt.colorbar(cf, ax=axes[1, 1], orientation="horizontal", pad=0.03, aspect=30, shrink=0.85, label="deg F")
    axes[1, 1].set_title("Dewpoint Correction  (Meso - HRRR)", fontsize=14, fontweight="bold")

    fig.suptitle(
        f"{title_prefix}  |  {analysis_time:%Y-%m-%d %H:%M UTC}  |  "
        f"{len(obs_qc):,} obs (3-pass QC)",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close(fig)


tag = analysis_time.strftime("%Y%m%d_%H%M")
from pathlib import Path
Path(f"output/compare_{tag}").mkdir(parents=True, exist_ok=True)

make_4panel([-125, -66, 24, 50], "Raw HRRR vs Mesoanalysis", f"output/compare_{tag}/conus.png", obs_size=2)
make_4panel([-105, -90, 28, 42], "Southern Plains Zoom", f"output/compare_{tag}/plains.png", obs_size=10)
make_4panel([-95, -75, 25, 38], "Southeast Zoom", f"output/compare_{tag}/southeast.png", obs_size=10)
