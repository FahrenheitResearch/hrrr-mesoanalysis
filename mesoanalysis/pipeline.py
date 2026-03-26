"""Full mesoanalysis pipeline orchestrator."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mesoanalysis.config import BARNES_CONFIGS, BARNES


def run(analysis_time, output_dir=None, fxx=0, model="hrrr", render_static=True):
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
    render_static : bool
        If True (default), render matplotlib PNGs.  Set False to skip
        static rendering and only produce web overlays (much faster).
    """
    model = model.lower()

    if output_dir is None:
        output_dir = Path(f"./output/{model}/{analysis_time:%Y%m%d_%H%M}")

    # Select Barnes parameters for this model
    barnes_cfg = BARNES_CONFIGS.get(model, BARNES)
    model_name = model.upper()

    print(f"=== Mesoanalysis for {analysis_time} ({model_name}) ===")

    # 1. Load model data
    print(f"\n[1/9] Loading {model_name} data ...")
    from .ingest import load_model
    model_data = load_model(analysis_time, model=model, fxx=fxx)

    # 2. Fetch observations (multi-source: ASOS + mesonets + NWS + NDBC + CWOP)
    print("[2/9] Fetching surface observations (multi-source) ...")
    from .obs.fetch_multi import fetch_all_surface_obs
    obs = fetch_all_surface_obs(analysis_time, window_minutes=20)
    print(f"       {len(obs)} observations fetched (multi-source)")

    # 3. QC observations against model background
    print("[3/9] Quality-controlling observations ...")
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

    # 4. Barnes objective analysis -- run all 5 fields in parallel
    #    (Barnes uses numpy/scipy which release the GIL during heavy computation)
    print("[4/9] Running Barnes objective analysis (parallel) ...")
    from .analysis.barnes import barnes_analysis

    obs_lats = obs_qc.lats
    obs_lons = obs_qc.lons
    n_all = len(obs_qc)

    # MSLP uses a different subset of observations
    obs_pres = obs_qc.with_pressure()
    print(f"       mslp: using {len(obs_pres)} of {n_all} obs (pressure-valid)")

    def _run_barnes(name, background, o_lats, o_lons, o_values):
        """Run a single Barnes analysis and return (name, result)."""
        result = barnes_analysis(
            model_data.lat, model_data.lon, background,
            o_lats, o_lons, o_values,
            kappa=barnes_cfg.kappa, gamma=barnes_cfg.gamma,
            passes=barnes_cfg.passes,
        )
        return name, result

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(_run_barnes, "t2m", t2m_C,
                        obs_lats, obs_lons, obs_qc.t_array),
            pool.submit(_run_barnes, "td2m", td2m_C,
                        obs_lats, obs_lons, obs_qc.td_array),
            pool.submit(_run_barnes, "u10", model_data.u10,
                        obs_lats, obs_lons, obs_qc.u_array),
            pool.submit(_run_barnes, "v10", model_data.v10,
                        obs_lats, obs_lons, obs_qc.v_array),
            pool.submit(_run_barnes, "mslp", mslp_hPa,
                        obs_pres.lats, obs_pres.lons, obs_pres.mslp_array),
        ]
        barnes_results = {}
        for f in futures:
            name, result = f.result()
            barnes_results[name] = result
            print(f"       {name} done")

    analyzed_t2m = barnes_results["t2m"]
    analyzed_td2m = barnes_results["td2m"]
    analyzed_u10 = barnes_results["u10"]
    analyzed_v10 = barnes_results["v10"]
    analyzed_mslp = barnes_results["mslp"]

    # 5. Merge analyzed surface into model data
    print(f"[5/9] Merging analysis into {model_name} grid ...")
    from .analysis.fields import merge_analysis
    merged = merge_analysis(
        model_data, analyzed_t2m, analyzed_td2m, analyzed_u10, analyzed_v10, analyzed_mslp,
    )

    # 6. Compute parameters
    print("[6/9] Computing thermodynamic parameters ...")
    from .params.thermodynamic import (
        compute_cape_fields, compute_theta_e, compute_wet_bulb,
        compute_lapse_rates, compute_mixing_ratio, compute_pw,
    )
    from .params.kinematic import (
        compute_shear_fields, compute_srh_fields, compute_bunkers_storm_motion,
    )
    from .params.composite import (
        compute_composite_fields, compute_ehi, compute_ship, compute_dcape,
        compute_dcp, compute_enhanced_scp, compute_critical_angle,
    )
    from .params.indices import compute_stability_indices, compute_lifted_index
    from .params.surface import (
        compute_heat_index, compute_windchill, compute_fosberg_ffwi,
        compute_hot_dry_windy, compute_wbgt,
    )

    thermo = {}
    thermo.update(compute_cape_fields(merged))
    thermo.update(compute_theta_e(merged))
    thermo.update(compute_wet_bulb(merged))
    thermo.update(compute_lapse_rates(merged))
    thermo.update(compute_mixing_ratio(merged))
    thermo.update(compute_pw(merged))

    print("[7/9] Computing kinematic & composite parameters ...")
    kinematic = {}
    kinematic.update(compute_shear_fields(merged))
    kinematic.update(compute_srh_fields(merged))

    # Bunkers storm motion (column-by-column, may be slow on full grid)
    print("       Computing Bunkers storm motion ...")
    kinematic.update(compute_bunkers_storm_motion(merged))

    # Basic composites (STP, SCP)
    composites = compute_composite_fields(merged, thermo, kinematic)

    # EHI
    composites.update(compute_ehi(thermo, kinematic))

    # SHIP
    print("       Computing SHIP ...")
    composites.update(compute_ship(merged, thermo, kinematic))

    # Enhanced SCP
    print("       Computing enhanced SCP ...")
    composites.update(compute_enhanced_scp(merged, thermo, kinematic))

    # DCAPE (column-by-column)
    print("       Computing DCAPE ...")
    dcape_result = compute_dcape(merged)
    composites.update(dcape_result)

    # DCP (needs DCAPE)
    print("       Computing DCP ...")
    composites.update(compute_dcp(merged, thermo, kinematic, dcape_result["dcape"]))

    # Critical angle (needs Bunkers)
    print("       Computing critical angle ...")
    composites.update(compute_critical_angle(merged, kinematic))

    # Stability indices (K-Index, Total Totals, Boyden, SWEAT)
    print("       Computing stability indices ...")
    composites.update(compute_stability_indices(merged))

    # Lifted Index (column-by-column)
    print("       Computing lifted index ...")
    composites.update(compute_lifted_index(merged))

    print("[8/9] Computing surface parameters ...")
    surface = {}
    surface.update(compute_heat_index(merged))
    surface.update(compute_windchill(merged))
    surface.update(compute_fosberg_ffwi(merged))
    surface.update(compute_hot_dry_windy(merged))
    surface.update(compute_wbgt(merged))

    # 7. Build full params dict for plotting
    params = {}
    params.update(thermo)
    params.update(kinematic)
    params.update(composites)
    params.update(surface)

    # Add surface fields
    params["t2m_f"] = (merged.t2m_K - 273.15) * 9 / 5 + 32
    params["td2m_f"] = (merged.td2m_K - 273.15) * 9 / 5 + 32

    # 8. Render static maps (matplotlib PNGs) -- optional
    if render_static:
        print("[9/10] Rendering static maps (parallel) ...")
        from .plotting.render import render_all

        render_all(merged.lon, merged.lat, params, analysis_time, output_dir,
                   model_name=model_name)
    else:
        print("[9/10] Skipping static map rendering (render_static=False)")

    # 9. Export web overlays (transparent PNGs + manifest + grids)
    print("[10/10] Exporting web overlays ...")
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
