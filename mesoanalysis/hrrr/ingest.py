"""HRRR data ingest via Herbie.

Loads surface and pressure-level fields, returns a single `HRRRData`
dataclass that downstream modules consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
from herbie import Herbie

from mesoanalysis.config import HRRR_PRESSURE_LEVELS_MB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HRRRData:
    """All HRRR fields needed by the analysis system.

    Surface arrays have shape (ny, nx).
    3-D arrays have shape (nz, ny, nx) where nz = len(levels_mb).
    """

    # Grid geometry
    lat: np.ndarray          # (ny, nx) float64
    lon: np.ndarray          # (ny, nx) float64
    ny: int
    nx: int

    # Surface fields
    t2m_K: np.ndarray        # 2-m temperature [K]
    td2m_K: np.ndarray       # 2-m dewpoint [K]
    u10: np.ndarray          # 10-m U wind [m/s]
    v10: np.ndarray          # 10-m V wind [m/s]
    psfc_Pa: np.ndarray      # Surface pressure [Pa]
    sfc_hgt: np.ndarray      # Surface geopotential height [m]
    mslp_Pa: np.ndarray      # Mean sea-level pressure [Pa]
    refc_dbz: np.ndarray     # Composite reflectivity [dBZ]
    cape_sfc: np.ndarray     # Surface-based CAPE [J/kg]

    # 3-D fields  (nz, ny, nx)
    t_C: np.ndarray          # Temperature [C]
    q_kgkg: np.ndarray       # Mixing ratio [kg/kg]
    h_agl_m: np.ndarray      # Height AGL [m]
    p_Pa: np.ndarray         # Pressure [Pa]
    u: np.ndarray            # U wind [m/s]
    v: np.ndarray            # V wind [m/s]

    # Metadata
    levels_mb: List[float] = field(default_factory=list)
    date: Optional[datetime] = None
    fxx: int = 0


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


def _extract_latlon(ds) -> tuple[np.ndarray, np.ndarray]:
    """Pull latitude/longitude 2-D arrays from a Herbie xarray Dataset."""
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    return lat, lon


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_hrrr(dt: datetime, fxx: int = 0) -> HRRRData:
    """Fetch HRRR surface + pressure-level data and return `HRRRData`.

    Parameters
    ----------
    dt : datetime
        Model initialisation time (UTC).
    fxx : int
        Forecast hour (0 for analysis).

    Returns
    -------
    HRRRData
    """
    dt_str = dt.strftime("%Y-%m-%d %H:%M")
    logger.info("Loading HRRR  init=%s  fxx=%d", dt_str, fxx)

    H_sfc = Herbie(dt_str, model="hrrr", fxx=fxx, product="sfc")
    H_prs = Herbie(dt_str, model="hrrr", fxx=fxx, product="prs")

    # ---- Surface fields ---------------------------------------------------
    ds_t2m = H_sfc.xarray(":TMP:2 m above ground")
    lat, lon = _extract_latlon(ds_t2m)
    ny, nx = lat.shape
    t2m_K = _first_var(ds_t2m).astype(np.float64)

    ds_td = H_sfc.xarray(":DPT:2 m above ground")
    td2m_K = _first_var(ds_td).astype(np.float64)

    ds_u10 = H_sfc.xarray(":UGRD:10 m above ground")
    u10 = _first_var(ds_u10).astype(np.float64)

    ds_v10 = H_sfc.xarray(":VGRD:10 m above ground")
    v10 = _first_var(ds_v10).astype(np.float64)

    ds_psfc = H_sfc.xarray(":PRES:surface")
    psfc_Pa = _first_var(ds_psfc).astype(np.float64)

    ds_hgt = H_sfc.xarray(":HGT:surface")
    sfc_hgt = _first_var(ds_hgt).astype(np.float64)

    ds_mslp = H_sfc.xarray(":MSLMA:mean sea level")
    mslp_Pa = _first_var(ds_mslp).astype(np.float64)

    ds_refc = H_sfc.xarray(":REFC:")
    refc_dbz = _first_var(ds_refc).astype(np.float64)

    ds_cape = H_sfc.xarray(":CAPE:surface")
    cape_sfc = _first_var(ds_cape).astype(np.float64)

    logger.info("Surface fields loaded  (%d x %d)", ny, nx)

    # ---- Pressure-level fields --------------------------------------------
    levels = HRRR_PRESSURE_LEVELS_MB
    nz = len(levels)

    t_C = np.empty((nz, ny, nx), dtype=np.float64)
    q_kgkg = np.empty((nz, ny, nx), dtype=np.float64)
    h_agl_m = np.empty((nz, ny, nx), dtype=np.float64)
    p_Pa = np.empty((nz, ny, nx), dtype=np.float64)
    u_3d = np.empty((nz, ny, nx), dtype=np.float64)
    v_3d = np.empty((nz, ny, nx), dtype=np.float64)

    for k, lev in enumerate(levels):
        lev_int = int(lev) if lev == int(lev) else lev
        lev_str = str(lev_int) if isinstance(lev_int, int) else str(lev)
        logger.debug("  pressure level %s mb", lev_str)

        # Temperature
        ds_t = H_prs.xarray(f":TMP:{lev_str} mb")
        t_k = _first_var(ds_t)
        t_C[k] = (t_k - 273.15).astype(np.float64)

        # Dewpoint -> mixing ratio
        ds_td = H_prs.xarray(f":DPT:{lev_str} mb")
        td_k = _first_var(ds_td)
        q_kgkg[k] = _mixing_ratio_from_dewpoint(td_k, lev)

        # Geopotential height -> height AGL
        ds_h = H_prs.xarray(f":HGT:{lev_str} mb")
        hgt_msl = _first_var(ds_h).astype(np.float64)
        h_agl_m[k] = hgt_msl - sfc_hgt

        # Pressure (constant across the slab)
        p_Pa[k] = lev * 100.0

        # Winds
        ds_u = H_prs.xarray(f":UGRD:{lev_str} mb")
        u_3d[k] = _first_var(ds_u).astype(np.float64)

        ds_v = H_prs.xarray(f":VGRD:{lev_str} mb")
        v_3d[k] = _first_var(ds_v).astype(np.float64)

    logger.info("Pressure-level fields loaded  (%d levels)", nz)

    return HRRRData(
        lat=lat,
        lon=lon,
        ny=ny,
        nx=nx,
        t2m_K=t2m_K,
        td2m_K=td2m_K,
        u10=u10,
        v10=v10,
        psfc_Pa=psfc_Pa,
        sfc_hgt=sfc_hgt,
        mslp_Pa=mslp_Pa,
        refc_dbz=refc_dbz,
        cape_sfc=cape_sfc,
        t_C=t_C,
        q_kgkg=q_kgkg,
        h_agl_m=h_agl_m,
        p_Pa=p_Pa,
        u=u_3d,
        v=v_3d,
        levels_mb=list(levels),
        date=dt,
        fxx=fxx,
    )
