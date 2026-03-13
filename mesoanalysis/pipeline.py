"""Full mesoanalysis pipeline orchestrator."""

from pathlib import Path

from mesoanalysis.config import BARNES


def run(analysis_time, output_dir=None, fxx=0):
    """Run full mesoanalysis pipeline."""
    if output_dir is None:
        output_dir = Path(f"./output/{analysis_time:%Y%m%d_%H%M}")

    print(f"=== Mesoanalysis for {analysis_time} ===")

    # 1. Load HRRR
    print("\n[1/8] Loading HRRR data ...")
    from .hrrr.ingest import load_hrrr
    hrrr = load_hrrr(analysis_time, fxx=fxx)

    # 2. Fetch observations
    print("[2/8] Fetching surface observations ...")
    from .obs.fetch import fetch_surface_obs
    obs = fetch_surface_obs(analysis_time, window_minutes=20)
    print(f"       {len(obs)} observations fetched")

    # 3. QC observations against HRRR background
    print("[3/8] Quality-controlling observations ...")
    from .obs.qc import qc_obs, HRRRFirstGuess

    # Build HRRRFirstGuess from HRRRData — need monotonically increasing
    # 1-D lat/lon for RegularGridInterpolator.
    lats_2d = hrrr.lat
    lons_2d = hrrr.lon

    # HRRR is curvilinear but for QC interpolation we approximate with
    # the first row (lons) and first column (lats).
    lats_1d = lats_2d[:, 0]
    lons_1d = lons_2d[0, :]

    # Ensure monotonically increasing
    if lats_1d[0] > lats_1d[-1]:
        lats_1d = lats_1d[::-1]
        flip_lat = True
    else:
        flip_lat = False

    if lons_1d[0] > lons_1d[-1]:
        lons_1d = lons_1d[::-1]
        flip_lon = True
    else:
        flip_lon = False

    def _orient(arr):
        """Flip array axes to match sorted lat/lon."""
        out = arr
        if flip_lat:
            out = out[::-1, :]
        if flip_lon:
            out = out[:, ::-1]
        return out

    # Convert units for QC: HRRRData stores K and Pa
    t2m_C = hrrr.t2m_K - 273.15
    td2m_C = hrrr.td2m_K - 273.15
    mslp_hPa = hrrr.mslp_Pa / 100.0

    first_guess = HRRRFirstGuess(
        lats_1d=lats_1d,
        lons_1d=lons_1d,
        t_2m=_orient(t2m_C),
        td_2m=_orient(td2m_C),
        u_10m=_orient(hrrr.u10),
        v_10m=_orient(hrrr.v10),
        mslp=_orient(mslp_hPa),
    )

    obs_qc = qc_obs(obs, first_guess)
    print(f"       {len(obs_qc)} observations passed QC")

    # 4. Barnes objective analysis — one call per surface field
    print("[4/8] Running Barnes objective analysis ...")
    from .analysis.barnes import barnes_analysis

    obs_lats = obs_qc.lats
    obs_lons = obs_qc.lons

    analyzed_t2m = barnes_analysis(
        hrrr.lat, hrrr.lon, t2m_C,
        obs_lats, obs_lons, obs_qc.t_array,
        kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes,
    )
    print("       t2m done")

    analyzed_td2m = barnes_analysis(
        hrrr.lat, hrrr.lon, td2m_C,
        obs_lats, obs_lons, obs_qc.td_array,
        kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes,
    )
    print("       td2m done")

    analyzed_u10 = barnes_analysis(
        hrrr.lat, hrrr.lon, hrrr.u10,
        obs_lats, obs_lons, obs_qc.u_array,
        kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes,
    )
    print("       u10 done")

    analyzed_v10 = barnes_analysis(
        hrrr.lat, hrrr.lon, hrrr.v10,
        obs_lats, obs_lons, obs_qc.v_array,
        kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes,
    )
    print("       v10 done")

    analyzed_mslp = barnes_analysis(
        hrrr.lat, hrrr.lon, mslp_hPa,
        obs_lats, obs_lons, obs_qc.mslp_array,
        kappa=BARNES.kappa, gamma=BARNES.gamma, passes=BARNES.passes,
    )
    print("       mslp done")

    # 5. Merge analyzed surface into HRRR
    print("[5/8] Merging analysis into HRRR grid ...")
    from .analysis.fields import merge_analysis
    merged = merge_analysis(
        hrrr, analyzed_t2m, analyzed_td2m, analyzed_u10, analyzed_v10, analyzed_mslp,
    )

    # 6. Compute parameters
    print("[6/8] Computing thermodynamic parameters ...")
    from .params.thermodynamic import (
        compute_cape_fields, compute_theta_e, compute_wet_bulb,
        compute_lapse_rates, compute_mixing_ratio, compute_pw,
    )
    from .params.kinematic import compute_shear_fields, compute_srh_fields
    from .params.composite import compute_composite_fields

    thermo = {}
    thermo.update(compute_cape_fields(merged))
    thermo.update(compute_theta_e(merged))
    thermo.update(compute_wet_bulb(merged))
    thermo.update(compute_lapse_rates(merged))
    thermo.update(compute_mixing_ratio(merged))
    thermo.update(compute_pw(merged))

    print("[7/8] Computing kinematic & composite parameters ...")
    kinematic = {}
    kinematic.update(compute_shear_fields(merged))
    kinematic.update(compute_srh_fields(merged))
    composites = compute_composite_fields(merged, thermo, kinematic)

    # 7. Build full params dict for plotting
    params = {}
    params.update(thermo)
    params.update(kinematic)
    params.update(composites)

    # Add surface fields
    params["t2m_f"] = (merged.t2m_K - 273.15) * 9 / 5 + 32
    params["td2m_f"] = (merged.td2m_K - 273.15) * 9 / 5 + 32

    # 8. Render all maps
    print("[8/8] Rendering maps ...")
    from .plotting.render import render_all

    render_all(merged.lon, merged.lat, params, analysis_time, output_dir)

    print(f"\n=== Done. Output in {output_dir} ===")
    return params
