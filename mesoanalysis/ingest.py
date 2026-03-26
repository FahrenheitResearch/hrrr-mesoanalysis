"""Generic NWP model ingest via Herbie.

Loads surface and pressure-level fields for any supported model (HRRR, RAP,
NAM) and returns a single `ModelData` dataclass that downstream modules
consume.

Handles models where surface and pressure fields share the same GRIB product
(RAP, NAM) by batching fields into single `.xarray()` calls to avoid
Herbie's subset-file overwrite issue.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

import numpy as np
from herbie import Herbie

from mesoanalysis.config import MODEL_CONFIGS, ModelConfig
from mesoanalysis.models import ModelData

logger = logging.getLogger(__name__)

# Maximum parallel downloads — avoids overwhelming NOMADS / AWS
_MAX_DOWNLOAD_WORKERS = 6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_var(ds):
    """Return the first data-variable array from an xarray Dataset."""
    key = list(ds.data_vars)[0]
    return ds[key].values


def _mixing_ratio_from_dewpoint(td_K: np.ndarray, lev_hPa: float) -> np.ndarray:
    """Compute mixing ratio (kg/kg) from dewpoint using Bolton (1980)."""
    td_C = td_K - 273.15
    es = 6.112 * np.exp(17.67 * td_C / (td_C + 243.5))  # hPa
    q = 0.622 * es / (lev_hPa - es)  # kg/kg
    return q.astype(np.float64)


def _mixing_ratio_from_rh(rh_frac: np.ndarray, t_K: np.ndarray, lev_hPa: float) -> np.ndarray:
    """Compute mixing ratio (kg/kg) from RH and temperature.

    Used for NAM which provides RH at pressure levels instead of DPT.
    Bolton (1980) saturation vapour pressure formula.

    Parameters
    ----------
    rh_frac : ndarray
        Relative humidity as a fraction (0-1).  If values > 2 are detected
        they are assumed to be percentages and divided by 100.
    t_K : ndarray
        Temperature in Kelvin.
    lev_hPa : float
        Pressure level in hPa.
    """
    # Auto-detect percent vs fraction
    rh = rh_frac.astype(np.float64)
    if np.nanmax(rh) > 2.0:
        rh = rh / 100.0

    t_C = t_K - 273.15
    es = 6.112 * np.exp(17.67 * t_C / (t_C + 243.5))  # hPa
    e = rh * es
    q = 0.622 * e / (lev_hPa - e)  # kg/kg
    return np.maximum(q, 0.0).astype(np.float64)


def _extract_latlon(ds) -> tuple[np.ndarray, np.ndarray]:
    """Pull latitude/longitude 2-D arrays from a Herbie xarray Dataset."""
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    return lat, lon


def _find_var(ds_list, *names: str) -> Optional[np.ndarray]:
    """Search a list of xarray Datasets for a variable by candidate names.

    Herbie's `.xarray()` with cfgrib often returns a list of Datasets (one
    per "hypercube").  Variable short-names can vary across models and
    versions, so we try several candidates.

    Returns the values of the first match found, or None.
    """
    for ds in ds_list:
        for name in names:
            if name in ds.data_vars:
                return ds[name].values
    return None


def _find_var_or_first(ds_list, *names: str) -> np.ndarray:
    """Like _find_var but falls back to returning the first data variable
    from the first dataset if no candidate name matches."""
    result = _find_var(ds_list, *names)
    if result is not None:
        return result
    # Fallback: first data variable from first dataset
    for ds in ds_list:
        if ds.data_vars:
            return _first_var(ds)
    raise ValueError("No data variables found in any dataset")


def _find_latlon(ds_list) -> tuple[np.ndarray, np.ndarray]:
    """Extract lat/lon from the first dataset in a list that has them."""
    for ds in ds_list:
        if "latitude" in ds.coords or "latitude" in ds:
            return _extract_latlon(ds)
    raise ValueError("No dataset contains latitude/longitude coordinates")


def _ensure_list(result) -> list:
    """Ensure Herbie .xarray() result is always a list of Datasets.

    Herbie may return a single Dataset or a list of Datasets depending on
    how many cfgrib hypercubes are found.
    """
    if isinstance(result, list):
        return result
    return [result]


# ---------------------------------------------------------------------------
# Separate-product loader (HRRR-style: sfc_product != prs_product)
# ---------------------------------------------------------------------------

def _load_separate_products(
    dt_str: str, cfg: ModelConfig, fxx: int, model: str,
) -> ModelData:
    """Load model data when surface and pressure fields are in different
    GRIB products (e.g. HRRR sfc vs prs).  Each .xarray() call targets a
    different product file so there is no subset-file collision."""

    def _H(product):
        return Herbie(dt_str, model=cfg.herbie_model, fxx=fxx, product=product)

    # ---- Surface fields ---------------------------------------------------
    ds_t2m = _H(cfg.sfc_product).xarray(":TMP:2 m above ground")
    lat, lon = _extract_latlon(ds_t2m)
    ny, nx = lat.shape
    t2m_K = _first_var(ds_t2m).astype(np.float64)

    ds_td = _H(cfg.sfc_product).xarray(":DPT:2 m above ground")
    td2m_K = _first_var(ds_td).astype(np.float64)

    ds_u10 = _H(cfg.sfc_product).xarray(":UGRD:10 m above ground")
    u10 = _first_var(ds_u10).astype(np.float64)

    ds_v10 = _H(cfg.sfc_product).xarray(":VGRD:10 m above ground")
    v10 = _first_var(ds_v10).astype(np.float64)

    ds_psfc = _H(cfg.sfc_product).xarray(":PRES:surface")
    psfc_Pa = _first_var(ds_psfc).astype(np.float64)

    ds_hgt = _H(cfg.sfc_product).xarray(":HGT:surface")
    sfc_hgt = _first_var(ds_hgt).astype(np.float64)

    ds_mslp = _H(cfg.sfc_product).xarray(cfg.mslp_search)
    mslp_Pa = _first_var(ds_mslp).astype(np.float64)

    try:
        ds_refc = _H(cfg.sfc_product).xarray(cfg.refc_search)
        refc_dbz = _first_var(ds_refc).astype(np.float64)
    except Exception:
        logger.warning(
            "%s: REFC field not available (%s), filling with zeros.",
            model.upper(), cfg.refc_search,
        )
        refc_dbz = np.zeros((ny, nx), dtype=np.float64)

    try:
        ds_cape = _H(cfg.sfc_product).xarray(":CAPE:surface")
        cape_sfc = _first_var(ds_cape).astype(np.float64)
    except Exception:
        logger.warning(
            "%s: CAPE:surface field not available, filling with zeros.",
            model.upper(),
        )
        cape_sfc = np.zeros((ny, nx), dtype=np.float64)

    # Downward shortwave radiation (for WBGT)
    dswrf_wm2 = None
    try:
        ds_dswrf = _H(cfg.sfc_product).xarray(":DSWRF:surface")
        dswrf_wm2 = _first_var(ds_dswrf).astype(np.float64)
    except Exception:
        logger.warning(
            "%s: DSWRF:surface not available, WBGT will not be computed.",
            model.upper(),
        )

    # Total cloud cover
    tcdc_pct = None
    try:
        ds_tcdc = _H(cfg.sfc_product).xarray(":TCDC:entire atmosphere")
        tcdc_pct = _first_var(ds_tcdc).astype(np.float64)
    except Exception:
        logger.warning(
            "%s: TCDC:entire atmosphere not available.",
            model.upper(),
        )

    logger.info("Surface fields loaded  (%d x %d)", ny, nx)

    # ---- Pressure-level fields (parallel downloads) -----------------------
    levels = cfg.pressure_levels
    nz = len(levels)

    t_C = np.empty((nz, ny, nx), dtype=np.float64)
    q_kgkg = np.empty((nz, ny, nx), dtype=np.float64)
    h_agl_m = np.empty((nz, ny, nx), dtype=np.float64)
    p_Pa = np.empty((nz, ny, nx), dtype=np.float64)
    u_3d = np.empty((nz, ny, nx), dtype=np.float64)
    v_3d = np.empty((nz, ny, nx), dtype=np.float64)

    def _fetch_level_separate(k, lev):
        """Fetch all fields for one pressure level (separate-product mode).

        Each H_prs.xarray() downloads a distinct subset file, so parallel
        calls are safe.
        """
        lev_int = int(lev) if lev == int(lev) else lev
        lev_str = str(lev_int) if isinstance(lev_int, int) else str(lev)
        logger.debug("  pressure level %s mb", lev_str)

        ds_t = _H(cfg.prs_product).xarray(f":TMP:{lev_str} mb")
        t_k = _first_var(ds_t)
        t_c = (t_k - 273.15).astype(np.float64)

        if cfg.has_dewpoint_prs:
            ds_td = _H(cfg.prs_product).xarray(f":DPT:{lev_str} mb")
            td_k = _first_var(ds_td)
            q = _mixing_ratio_from_dewpoint(td_k, lev)
        else:
            ds_rh = _H(cfg.prs_product).xarray(f":RH:{lev_str} mb")
            rh_vals = _first_var(ds_rh)
            q = _mixing_ratio_from_rh(rh_vals, t_k, lev)

        ds_h = _H(cfg.prs_product).xarray(f":HGT:{lev_str} mb")
        hgt_msl = _first_var(ds_h).astype(np.float64)
        h_agl = hgt_msl - sfc_hgt

        ds_u = _H(cfg.prs_product).xarray(f":UGRD:{lev_str} mb")
        u_lev = _first_var(ds_u).astype(np.float64)

        ds_v = _H(cfg.prs_product).xarray(f":VGRD:{lev_str} mb")
        v_lev = _first_var(ds_v).astype(np.float64)

        return k, t_c, q, h_agl, lev * 100.0, u_lev, v_lev

    with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_level_separate, k, lev): k
            for k, lev in enumerate(levels)
        }
        for future in as_completed(futures):
            k, t_c, q, h_agl, p_val, u_lev, v_lev = future.result()
            t_C[k] = t_c
            q_kgkg[k] = q
            h_agl_m[k] = h_agl
            p_Pa[k] = p_val
            u_3d[k] = u_lev
            v_3d[k] = v_lev

    logger.info("Pressure-level fields loaded  (%d levels)", nz)

    return ModelData(
        lat=lat, lon=lon, ny=ny, nx=nx,
        t2m_K=t2m_K, td2m_K=td2m_K, u10=u10, v10=v10,
        psfc_Pa=psfc_Pa, sfc_hgt=sfc_hgt, mslp_Pa=mslp_Pa,
        refc_dbz=refc_dbz, cape_sfc=cape_sfc,
        t_C=t_C, q_kgkg=q_kgkg, h_agl_m=h_agl_m, p_Pa=p_Pa,
        u=u_3d, v=v_3d,
        dswrf_wm2=dswrf_wm2, tcdc_pct=tcdc_pct,
        levels_mb=list(levels), date=None, fxx=fxx,
        model=model, resolution_km=cfg.resolution_km,
    )


# ---------------------------------------------------------------------------
# Same-product loader (RAP / NAM style: sfc_product == prs_product)
# ---------------------------------------------------------------------------

def _build_sfc_search(cfg: ModelConfig) -> str:
    """Build a combined regex search string for all surface fields.

    This lets us download one GRIB subset containing every surface field
    we need, avoiding Herbie's subset-file overwrite problem.
    """
    parts = [
        ":TMP:2 m above ground",
        ":DPT:2 m above ground",
        ":UGRD:10 m above ground",
        ":VGRD:10 m above ground",
        ":PRES:surface",
        ":HGT:surface",
        cfg.mslp_search,
        cfg.refc_search,
        ":CAPE:surface",
        ":DSWRF:surface",
        ":TCDC:entire atmosphere",
    ]
    return "(" + "|".join(parts) + ")"


def _build_level_search(lev_str: str, cfg: ModelConfig) -> str:
    """Build a combined regex for all fields at one pressure level.

    Downloads TMP, HGT, UGRD, VGRD, and the moisture variable (DPT or RH)
    for a single level in one .xarray() call.
    """
    parts = [
        f":TMP:{lev_str} mb",
        f":HGT:{lev_str} mb",
        f":UGRD:{lev_str} mb",
        f":VGRD:{lev_str} mb",
    ]
    if cfg.has_dewpoint_prs:
        parts.append(f":DPT:{lev_str} mb")
    else:
        parts.append(f":RH:{lev_str} mb")
    return "(" + "|".join(parts) + ")"


def _load_same_product(
    dt_str: str, cfg: ModelConfig, fxx: int, model: str,
) -> ModelData:
    """Load model data when surface and pressure fields share one GRIB
    product (e.g. RAP awp130pgrb, NAM conusnest.hiresf).

    Uses batched regex searches so each .xarray() call produces a unique
    subset file, preventing Herbie's overwrite issue.
    """

    def _H():
        return Herbie(dt_str, model=cfg.herbie_model, fxx=fxx, product=cfg.sfc_product)

    # ---- Surface fields (single batched download) -------------------------
    sfc_search = _build_sfc_search(cfg)
    logger.debug("Surface batch search: %s", sfc_search)

    sfc_ds_list = _ensure_list(_H().xarray(sfc_search))

    lat, lon = _find_latlon(sfc_ds_list)
    ny, nx = lat.shape

    # Extract surface variables by cfgrib short-name candidates
    t2m_raw = _find_var(sfc_ds_list, "t2m", "2t", "TMP_2maboveground")
    if t2m_raw is None:
        raise ValueError("Could not find 2-m temperature in surface batch")
    t2m_K = t2m_raw.astype(np.float64)

    td2m_raw = _find_var(sfc_ds_list, "d2m", "2d", "DPT_2maboveground")
    if td2m_raw is None:
        raise ValueError("Could not find 2-m dewpoint in surface batch")
    td2m_K = td2m_raw.astype(np.float64)

    u10_raw = _find_var(sfc_ds_list, "u10", "10u", "UGRD_10maboveground")
    if u10_raw is None:
        raise ValueError("Could not find 10-m U wind in surface batch")
    u10 = u10_raw.astype(np.float64)

    v10_raw = _find_var(sfc_ds_list, "v10", "10v", "VGRD_10maboveground")
    if v10_raw is None:
        raise ValueError("Could not find 10-m V wind in surface batch")
    v10 = v10_raw.astype(np.float64)

    psfc_raw = _find_var(sfc_ds_list, "sp", "surface_pressure", "PRES_surface")
    if psfc_raw is None:
        raise ValueError("Could not find surface pressure in surface batch")
    psfc_Pa = psfc_raw.astype(np.float64)

    sfc_hgt_raw = _find_var(sfc_ds_list, "orog", "z", "HGT_surface")
    if sfc_hgt_raw is None:
        raise ValueError("Could not find surface height in surface batch")
    sfc_hgt = sfc_hgt_raw.astype(np.float64)

    mslp_raw = _find_var(sfc_ds_list, "msl", "mslma", "prmsl", "mslet", "MSLMA", "PRMSL")
    if mslp_raw is None:
        raise ValueError("Could not find MSLP in surface batch")
    mslp_Pa = mslp_raw.astype(np.float64)

    refc_raw = _find_var(sfc_ds_list, "refc", "REFC", "refd", "unknown")
    if refc_raw is not None:
        refc_dbz = refc_raw.astype(np.float64)
    else:
        logger.warning(
            "%s: REFC not found in surface batch, filling with zeros.",
            model.upper(),
        )
        refc_dbz = np.zeros((ny, nx), dtype=np.float64)

    cape_raw = _find_var(sfc_ds_list, "cape", "CAPE", "CAPE_surface")
    if cape_raw is not None:
        cape_sfc = cape_raw.astype(np.float64)
    else:
        logger.warning(
            "%s: CAPE:surface not found in surface batch, filling with zeros.",
            model.upper(),
        )
        cape_sfc = np.zeros((ny, nx), dtype=np.float64)

    # Downward shortwave radiation (for WBGT)
    dswrf_raw = _find_var(sfc_ds_list, "dswrf", "DSWRF", "sdswrf")
    dswrf_wm2 = dswrf_raw.astype(np.float64) if dswrf_raw is not None else None
    if dswrf_wm2 is None:
        logger.warning(
            "%s: DSWRF not found in surface batch, WBGT will not be computed.",
            model.upper(),
        )

    # Total cloud cover
    tcdc_raw = _find_var(sfc_ds_list, "tcc", "TCDC", "tcdc")
    tcdc_pct = tcdc_raw.astype(np.float64) if tcdc_raw is not None else None
    if tcdc_pct is None:
        logger.warning(
            "%s: TCDC not found in surface batch.",
            model.upper(),
        )

    logger.info("Surface fields loaded  (%d x %d)", ny, nx)

    # ---- Pressure-level fields (parallel downloads) -----------------------
    levels = cfg.pressure_levels
    nz = len(levels)

    t_C = np.empty((nz, ny, nx), dtype=np.float64)
    q_kgkg = np.empty((nz, ny, nx), dtype=np.float64)
    h_agl_m = np.empty((nz, ny, nx), dtype=np.float64)
    p_Pa = np.empty((nz, ny, nx), dtype=np.float64)
    u_3d = np.empty((nz, ny, nx), dtype=np.float64)
    v_3d = np.empty((nz, ny, nx), dtype=np.float64)

    def _fetch_level_same(k, lev):
        """Fetch all fields for one pressure level (same-product mode).

        Each call uses a distinct batched regex that produces a unique
        subset file, so parallel calls are safe.
        """
        lev_int = int(lev) if lev == int(lev) else lev
        lev_str = str(lev_int) if isinstance(lev_int, int) else str(lev)
        logger.debug("  pressure level %s mb (batched)", lev_str)

        lev_search = _build_level_search(lev_str, cfg)
        lev_ds_list = _ensure_list(_H().xarray(lev_search))

        # Temperature
        t_k_raw = _find_var(lev_ds_list, "t", "TMP")
        if t_k_raw is None:
            t_k_raw = _find_var_or_first(lev_ds_list, "t")
        t_c = (t_k_raw - 273.15).astype(np.float64)

        # Moisture
        if cfg.has_dewpoint_prs:
            td_k_raw = _find_var(lev_ds_list, "dpt", "d", "DPT")
            if td_k_raw is None:
                td_k_raw = _find_var_or_first(lev_ds_list, "dpt")
            q = _mixing_ratio_from_dewpoint(td_k_raw, lev)
        else:
            rh_raw = _find_var(lev_ds_list, "r", "rh", "RH")
            if rh_raw is None:
                rh_raw = _find_var_or_first(lev_ds_list, "r")
            q = _mixing_ratio_from_rh(rh_raw, t_k_raw, lev)

        # Geopotential height -> height AGL
        hgt_raw = _find_var(lev_ds_list, "gh", "z", "HGT")
        if hgt_raw is None:
            hgt_raw = _find_var_or_first(lev_ds_list, "gh")
        h_agl = hgt_raw.astype(np.float64) - sfc_hgt

        # Winds
        u_raw = _find_var(lev_ds_list, "u", "UGRD")
        if u_raw is None:
            u_raw = _find_var_or_first(lev_ds_list, "u")
        u_lev = u_raw.astype(np.float64)

        v_raw = _find_var(lev_ds_list, "v", "VGRD")
        if v_raw is None:
            v_raw = _find_var_or_first(lev_ds_list, "v")
        v_lev = v_raw.astype(np.float64)

        return k, t_c, q, h_agl, lev * 100.0, u_lev, v_lev

    with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_level_same, k, lev): k
            for k, lev in enumerate(levels)
        }
        for future in as_completed(futures):
            k, t_c, q, h_agl, p_val, u_lev, v_lev = future.result()
            t_C[k] = t_c
            q_kgkg[k] = q
            h_agl_m[k] = h_agl
            p_Pa[k] = p_val
            u_3d[k] = u_lev
            v_3d[k] = v_lev

    logger.info("Pressure-level fields loaded  (%d levels)", nz)

    return ModelData(
        lat=lat, lon=lon, ny=ny, nx=nx,
        t2m_K=t2m_K, td2m_K=td2m_K, u10=u10, v10=v10,
        psfc_Pa=psfc_Pa, sfc_hgt=sfc_hgt, mslp_Pa=mslp_Pa,
        refc_dbz=refc_dbz, cape_sfc=cape_sfc,
        t_C=t_C, q_kgkg=q_kgkg, h_agl_m=h_agl_m, p_Pa=p_Pa,
        u=u_3d, v=v_3d,
        dswrf_wm2=dswrf_wm2, tcdc_pct=tcdc_pct,
        levels_mb=list(levels), date=None, fxx=fxx,
        model=model, resolution_km=cfg.resolution_km,
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_model(dt: datetime, model: str = "hrrr", fxx: int = 0) -> ModelData:
    """Fetch NWP model surface + pressure-level data and return `ModelData`.

    Parameters
    ----------
    dt : datetime
        Model initialisation time (UTC).
    model : str
        Model identifier: ``"hrrr"``, ``"rap"``, or ``"nam"``.
    fxx : int
        Forecast hour (0 for analysis).

    Returns
    -------
    ModelData
    """
    model = model.lower()
    if model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model {model!r}. Supported: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model]
    dt_str = dt.strftime("%Y-%m-%d %H:%M")
    logger.info("Loading %s  init=%s  fxx=%d", model.upper(), dt_str, fxx)

    same_product = cfg.sfc_product == cfg.prs_product

    if same_product:
        logger.info(
            "%s uses single product '%s' — batched download mode",
            model.upper(), cfg.sfc_product,
        )
        data = _load_same_product(dt_str, cfg, fxx, model)
    else:
        logger.info(
            "%s uses separate products (sfc=%s, prs=%s) — per-field mode",
            model.upper(), cfg.sfc_product, cfg.prs_product,
        )
        data = _load_separate_products(dt_str, cfg, fxx, model)

    # Stamp the datetime (both paths leave it as None)
    data.date = dt
    return data
