"""3-panel comparison: Raw HRRR | Mesoanalysis | Difference."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timezone, timedelta
from pathlib import Path

now = datetime.now(tz=timezone.utc)
analysis_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
tag = analysis_time.strftime('%Y%m%d_%H%M')
outdir = Path(f'output/compare3_{tag}')
outdir.mkdir(parents=True, exist_ok=True)

from mesoanalysis.hrrr.ingest import load_hrrr
hrrr = load_hrrr(analysis_time, fxx=0)

from mesoanalysis.obs.fetch_multi import fetch_all_surface_obs
from mesoanalysis.obs.qc import qc_obs, HRRRFirstGuess
from mesoanalysis.config import BARNES
from mesoanalysis.analysis.barnes import barnes_analysis
from mesoanalysis.analysis.fields import merge_analysis
from mesoanalysis.params.thermodynamic import compute_cape_fields, compute_theta_e, compute_mixing_ratio
from mesoanalysis.params.kinematic import compute_shear_fields, compute_srh_fields

obs = fetch_all_surface_obs(analysis_time, window_minutes=20)

t2m_C = hrrr.t2m_K - 273.15
td2m_C = hrrr.td2m_K - 273.15
mslp_hPa = hrrr.mslp_Pa / 100.0
lons_norm = hrrr.lon.copy()
lons_norm[lons_norm > 180] -= 360

first_guess = HRRRFirstGuess(
    lats_2d=hrrr.lat, lons_2d=lons_norm,
    t_2m=t2m_C, td_2m=td2m_C,
    u_10m=hrrr.u10, v_10m=hrrr.v10, mslp=mslp_hPa,
)
obs_qc = qc_obs(obs, first_guess)

# Barnes
analyzed_t2m = barnes_analysis(hrrr.lat, hrrr.lon, t2m_C, obs_qc.lats, obs_qc.lons, obs_qc.t_array, kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes)
analyzed_td2m = barnes_analysis(hrrr.lat, hrrr.lon, td2m_C, obs_qc.lats, obs_qc.lons, obs_qc.td_array, kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes)
analyzed_u10 = barnes_analysis(hrrr.lat, hrrr.lon, hrrr.u10, obs_qc.lats, obs_qc.lons, obs_qc.u_array, kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes)
analyzed_v10 = barnes_analysis(hrrr.lat, hrrr.lon, hrrr.v10, obs_qc.lats, obs_qc.lons, obs_qc.v_array, kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes)
obs_pres = obs_qc.with_pressure()
analyzed_mslp = barnes_analysis(hrrr.lat, hrrr.lon, mslp_hPa, obs_pres.lats, obs_pres.lons, obs_pres.mslp_array, kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes)

merged = merge_analysis(hrrr, analyzed_t2m, analyzed_td2m, analyzed_u10, analyzed_v10, analyzed_mslp)

# Compute derived params for both
raw_params = {}
raw_params.update(compute_cape_fields(hrrr))
raw_params.update(compute_theta_e(hrrr))
raw_params.update(compute_mixing_ratio(hrrr))

meso_params = {}
meso_params.update(compute_cape_fields(merged))
meso_params.update(compute_theta_e(merged))
meso_params.update(compute_mixing_ratio(merged))

# Surface fields
raw_params['t2m_f'] = (hrrr.t2m_K - 273.15) * 9/5 + 32
raw_params['td2m_f'] = (hrrr.td2m_K - 273.15) * 9/5 + 32
meso_params['t2m_f'] = analyzed_t2m * 9/5 + 32
meso_params['td2m_f'] = analyzed_td2m * 9/5 + 32

lon_plot = hrrr.lon.copy()
lon_plot[lon_plot > 180] -= 360
proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)


def tripanel(raw, meso, title, levels, cmap, fname, extent=[-125,-66,24,50],
             extend='max', diff_levels=None, diff_cmap='bwr', units=''):
    diff = meso - raw

    if diff_levels is None:
        mx = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        mx = min(mx, np.nanstd(diff) * 4)
        mx = max(mx, 1.0)
        diff_levels = np.linspace(-mx, mx, 17)

    fig, axes = plt.subplots(1, 3, figsize=(30, 9), subplot_kw={'projection': proj})

    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, edgecolor='#888', linewidth=0.4)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.6)

    # Raw
    cf1 = axes[0].contourf(lon_plot, hrrr.lat, raw, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend=extend)
    plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.03, aspect=30, shrink=0.9, label=units)
    axes[0].set_title('Raw HRRR', fontsize=15, fontweight='bold')

    # Meso
    cf2 = axes[1].contourf(lon_plot, hrrr.lat, meso, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend=extend)
    plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.03, aspect=30, shrink=0.9, label=units)
    axes[1].set_title('Mesoanalysis (HRRR + Obs)', fontsize=15, fontweight='bold')

    # Difference
    cf3 = axes[2].contourf(lon_plot, hrrr.lat, diff, levels=diff_levels, cmap=diff_cmap, transform=ccrs.PlateCarree(), extend='both')
    axes[2].scatter([o.lon for o in obs_qc], [o.lat for o in obs_qc], c='black', s=1.5, alpha=0.3, transform=ccrs.PlateCarree(), zorder=5)
    plt.colorbar(cf3, ax=axes[2], orientation='horizontal', pad=0.03, aspect=30, shrink=0.9, label=units)
    axes[2].set_title('Difference (Meso - HRRR)', fontsize=15, fontweight='bold')

    fig.suptitle(f'{title}  |  {analysis_time:%Y-%m-%d %H:%M UTC}  |  {len(obs_qc):,} obs', fontsize=17, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(outdir / fname), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'  {fname}')


print(f'Generating 3-panel comparisons -> {outdir}/')

# Surface fields
tripanel(raw_params['t2m_f'], meso_params['t2m_f'], '2m Temperature', np.arange(0, 96, 4), 'RdYlBu_r', 'temperature.png', extend='both', diff_levels=np.arange(-8, 8.5, 1), units='deg F')
tripanel(raw_params['td2m_f'], meso_params['td2m_f'], '2m Dewpoint', np.arange(0, 76, 4), 'BrBG', 'dewpoint.png', extend='both', diff_levels=np.arange(-8, 8.5, 1), units='deg F')

# Thermodynamic
tripanel(raw_params['sbcape'], meso_params['sbcape'], 'SBCAPE', [0,100,250,500,1000,1500,2000,2500,3000,4000,5000], 'YlOrRd', 'sbcape.png', diff_levels=np.arange(-500, 550, 50), units='J/kg')
tripanel(raw_params['mlcape'], meso_params['mlcape'], 'MLCAPE', [0,100,250,500,1000,1500,2000,2500,3000,4000], 'YlOrRd', 'mlcape.png', diff_levels=np.arange(-500, 550, 50), units='J/kg')
tripanel(raw_params['theta_e_sfc'], meso_params['theta_e_sfc'], 'Surface Theta-e', np.arange(280, 365, 5), 'RdYlBu_r', 'theta_e.png', extend='both', diff_levels=np.arange(-6, 6.5, 0.5), units='K')
tripanel(raw_params['mixing_ratio'], meso_params['mixing_ratio'], 'Mixing Ratio', np.arange(0, 22, 2), 'Greens', 'mixing_ratio.png', diff_levels=np.arange(-4, 4.5, 0.5), units='g/kg')

# Zoomed
tripanel(raw_params['t2m_f'], meso_params['t2m_f'], 'Temperature — Plains', np.arange(20, 85, 3), 'RdYlBu_r', 'temperature_plains.png', extent=[-105,-90,28,42], extend='both', diff_levels=np.arange(-8, 8.5, 1), units='deg F')
tripanel(raw_params['td2m_f'], meso_params['td2m_f'], 'Dewpoint — Plains', np.arange(10, 70, 3), 'BrBG', 'dewpoint_plains.png', extent=[-105,-90,28,42], extend='both', diff_levels=np.arange(-8, 8.5, 1), units='deg F')
tripanel(raw_params['sbcape'], meso_params['sbcape'], 'SBCAPE — Southeast', [0,100,250,500,1000,1500,2000,2500,3000,4000,5000], 'YlOrRd', 'sbcape_southeast.png', extent=[-95,-75,25,38], diff_levels=np.arange(-500, 550, 50), units='J/kg')

print(f'Done — {outdir}/')
