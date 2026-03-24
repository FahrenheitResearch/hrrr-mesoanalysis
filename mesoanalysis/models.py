"""NWP model data container.

Generic dataclass for model output consumed by the analysis system.
Supports HRRR, RAP, NAM, and any future NWP models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class ModelData:
    """All model fields needed by the analysis system.

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

    # Optional surface fields (may be None if unavailable)
    dswrf_wm2: Optional[np.ndarray] = None   # Downward SW radiation [W/m^2]
    tcdc_pct: Optional[np.ndarray] = None     # Total cloud cover [%]

    # Metadata
    levels_mb: List[float] = field(default_factory=list)
    date: Optional[datetime] = None
    fxx: int = 0
    model: str = "hrrr"
    resolution_km: float = 3.0
