"""Full mesoanalysis pipeline orchestrator."""

from pathlib import Path

from mesoanalysis.config import BARNES_CONFIGS, BARNES


def run(analysis_time, output_dir=None, fxx=0, model="hrrr"):
    """Run full mesoanalysis pipeline.

    Parameters
    ----------
    analysis_time : datetime
        Analysis valid time (UTC).
    output_dir : path-like, optional
        Root output directory.  Defaults to ``./output/{model}/{YYYYMMDD_HHMM}``.
    fxx : int
        Forecast hour (0 for analysis).
    model : str
        NWP model identifier: ``"hrrr"``, ``"rap"``, or ``"nam"``.
    """
    model = model.lower()

    if output_dir is None:
        output_dir = Path(f"./output/{model}/{analysis_time:%Y%m%d_%H%M}")

    # Select Barnes parameters for this model
    barnes_cfg = BARNES_CONFIGS.get(model, BARNES)
    model_name = model.upper()

    print(f"=== Mesoanalysis for {analysis_time} ({model_name}) ===")

    # 1. Load model data
    print(f"\n[1/8] Loading {model_name} data ...")
    from .ingest import load_model
    model_data = load_model(analysis_time, model=model, fxx=fxx)

    # 2. Fetch observations (multi-source: ASOS + mesonets + NWS + NDBC + CWOP)
    print("[2/8] Fetching surface observations (multi-source) ...")
    from .obs.fetch_multi import fetch_all_surface_obs
    obs = fetch_all_surface_obs(analysis_time, window_minutes=20)
    print(f"       {len(obs)} observations fetched (multi-source)")

    # 3. QC observations against model background
    print("[3/8] Quality-controlling observations ...")
    from .obs.qc import qc_obs, ModelFirstGuess

    # Convert units for QC: ModelData stores K and Pa
    t2m_C = model_data.t2m_K - 273.15
    td2m_C = model_data.td2m_K - 273.15
    mslp_hPa = model_data.mslp_Pa / 100.0

    # Normalize model longitudes from 0..360 to -180..180 to match obs
    lons_norm = model_data.lon.copy()
    lons_norm[lons_norm > 180] -= 360

    first_guess = ModelFirstGuess(
        lats_2d=model_data.lat,
        lons_2d=lons_norm,
        t_2m=t2m_C,
        td_2m=td2m_C,
        u_10m=model_data.u10,
        v_10m=model_data.v10,
        mslp=mslp_hPa,
    )

    obs_qc = qc_obs(obs, first_guess, model=model)
    print(f"       {len(obs_qc)} observations passed QC")

    # 4. Barnes objective analysis -- one call per surface field
    #    T, Td, u, v use ALL obs; mslp uses only obs with valid pressure
    print("[4/8] Running Barnes objective analysis ...")
    from .analysis.barnes import barnes_analysis

    obs_lats = obs_qc.lats
    obs_lons = obs_qc.lons
    n_all = len(obs_qc)

    analyzed_t2m = barnes_analysis(
        model_data.lat, model_data.lon, t2m_C,
        obs_lats, obs_lons, obs_qc.t_array,
        kappa=barnes_cfg.kappa, gamma=barnes_cfg.gamma, passes=barnes_cfg.passes,
    )
    print("       t2m done")

    analyzed_td2m = barnes_analysis(
        model_data.lat, model_data.lon, td2m_C,
        obs_lats, obs_lons, obs_qc.td_array,
        kappa=barnes_cfg.kappa, gamma=barnes_cfg.gamma, passes=barnes_cfg.passes,
    )
    print("       td2m done")

    analyzed_u10 = barnes_analysis(
        model_data.lat, model_data.lon, model_data.u10,
        obs_lats, obs_lons, obs_qc.u_array,
        kappa=barnes_cfg.kappa, gamma=barnes_cfg.gamma, passes=barnes_cfg.passes,
    )
    print("       u10 done")

    analyzed_v10 = barnes_analysis(
        model_data.lat, model_data.lon, model_data.v10,
        obs_lats, obs_lons, obs_qc.v_array,
        kappa=barnes_cfg.kappa, gamma=barnes_cfg.gamma, passes=barnes_cfg.passes,
    )
    print("       v10 done")

    # MSLP: only use obs that have valid pressure readings
    obs_pres = obs_qc.with_pressure()
    print(f"       mslp: using {len(obs_pres)} of {n_all} obs (pressure-valid)")
    analyzed_mslp = barnes_analysis(
        model_data.lat, model_data.lon, mslp_hPa,
        obs_pres.lats, obs_pres.lons, obs_pres.mslp_array,
        kappa=barnes_cfg.kappa, gamma=barnes_cfg.gamma, passes=barnes_cfg.passes,
    )
    print("       mslp done")

    # 5. Merge analyzed surface into model data
    print(f"[5/8] Merging analysis into {model_name} grid ...")
    from .analysis.fields import merge_analysis
    merged = merge_analysis(
        model_data, analyzed_t2m, analyzed_td2m, analyzed_u10, analyzed_v10, analyzed_mslp,
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

    # 8. Render static maps (matplotlib PNGs)
    print("[8/9] Rendering static maps ...")
    from .plotting.render import render_all

    render_all(merged.lon, merged.lat, params, analysis_time, output_dir,
               model_name=model_name)

    # 9. Export web overlays (transparent PNGs + manifest + grids)
    print("[9/9] Exporting web overlays ...")
    from .output.export import export_web_overlays

    export_web_overlays(
        params=params,
        lon=merged.lon,
        lat=merged.lat,
        analysis_time=analysis_time,
        output_dir=output_dir,
    )

    print(f"\n=== Done. Output in {output_dir} ===")
    return params
