"""Central configuration for the mesoscale analysis system."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cartopy.crs as ccrs


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainConfig:
    """CONUS domain extent and projection settings."""

    extent: Tuple[float, float, float, float] = (-125.0, -66.0, 24.0, 50.0)
    """(lon_min, lon_max, lat_min, lat_max)"""

    central_longitude: float = -97.5
    central_latitude: float = 38.5
    standard_parallels: Tuple[float, float] = (30.0, 60.0)

    @property
    def projection(self) -> ccrs.LambertConformal:
        return ccrs.LambertConformal(
            central_longitude=self.central_longitude,
            central_latitude=self.central_latitude,
            standard_parallels=self.standard_parallels,
        )

    @property
    def geodetic(self) -> ccrs.Geodetic:
        return ccrs.Geodetic()


# ---------------------------------------------------------------------------
# Observation matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObsConfig:
    """Observation time-matching parameters."""

    time_window_min: int = 20
    """Accept observations within +/- this many minutes of analysis time."""


# ---------------------------------------------------------------------------
# Barnes objective analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BarnesConfig:
    """Barnes successive-correction parameters."""

    kappa: float = 1.8e9
    """First-pass smoothing parameter (m^2).

    With ~1600 ASOS stations over CONUS, average spacing d ~ 70 km.
    Barnes (1964): kappa = (2d/pi)^2.  For d=70 km: kappa ~ 2.0e9.
    Using 1.8e9 gives cutoff = 4*sqrt(kappa) ~ 170 km, which keeps
    corrections localized while still providing smooth analysis fields.
    """

    gamma: float = 0.4
    """Convergence parameter applied on second pass (kappa2 = gamma * kappa).

    0.4 is tighter than the classic 0.3, giving better convergence toward
    observations on the second pass.
    """

    passes: int = 2
    """Number of analysis passes."""


# ---------------------------------------------------------------------------
# HRRR vertical profile
# ---------------------------------------------------------------------------

HRRR_PRESSURE_LEVELS_MB: List[float] = [
    1013.2, 1000, 975, 950, 925, 900, 875, 850,
    825, 800, 775, 750, 725, 700, 675, 650,
    625, 600, 575, 550, 525, 500,
]
"""Pressure levels (hPa) used to build 3-D HRRR profiles."""


# ---------------------------------------------------------------------------
# RAP / NAM vertical profiles
# ---------------------------------------------------------------------------

RAP_PRESSURE_LEVELS_MB: List[float] = [
    1000, 975, 950, 925, 900, 875, 850,
    825, 800, 775, 750, 725, 700, 675, 650,
    625, 600, 575, 550, 525, 500,
]
"""Pressure levels (hPa) used to build 3-D RAP profiles (no 1013.2)."""

NAM_PRESSURE_LEVELS_MB: List[float] = [
    1000, 975, 950, 925, 900, 875, 850,
    825, 800, 775, 750, 725, 700, 675, 650,
    625, 600, 575, 550, 525, 500,
]
"""Pressure levels (hPa) used to build 3-D NAM profiles (no 1013.2)."""


# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single NWP model."""

    herbie_model: str
    sfc_product: str
    prs_product: str
    pressure_levels: List[float]
    resolution_km: float
    mslp_search: str
    has_dewpoint_prs: bool
    refc_search: str


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "hrrr": ModelConfig(
        herbie_model="hrrr",
        sfc_product="sfc",
        prs_product="prs",
        pressure_levels=HRRR_PRESSURE_LEVELS_MB,
        resolution_km=3.0,
        mslp_search=":MSLMA:mean sea level",
        has_dewpoint_prs=True,
        refc_search=":REFC:",
    ),
    "rap": ModelConfig(
        herbie_model="rap",
        sfc_product="awp130pgrb",
        prs_product="awp130pgrb",
        pressure_levels=RAP_PRESSURE_LEVELS_MB,
        resolution_km=13.0,
        mslp_search=":MSLMA:mean sea level",
        has_dewpoint_prs=True,
        refc_search=":REFC:",
    ),
    "nam": ModelConfig(
        herbie_model="nam",
        sfc_product="conusnest.hiresf",
        prs_product="conusnest.hiresf",
        pressure_levels=NAM_PRESSURE_LEVELS_MB,
        resolution_km=5.0,
        mslp_search=":PRMSL:mean sea level",
        has_dewpoint_prs=False,
        refc_search=":REFC:",
    ),
}


# ---------------------------------------------------------------------------
# Per-model Barnes parameters
# ---------------------------------------------------------------------------

BARNES_CONFIGS: Dict[str, BarnesConfig] = {
    "hrrr": BarnesConfig(kappa=1.8e9, gamma=0.4),
    "rap": BarnesConfig(kappa=5.0e9, gamma=0.4),
    "nam": BarnesConfig(kappa=4.5e9, gamma=0.4),
}


# ---------------------------------------------------------------------------
# Per-model QC gross-error thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QCThresholds:
    """Gross-error thresholds for a given model."""

    t_C: float
    td_C: float
    wind_ms: float
    mslp_hPa: float


QC_THRESHOLDS: Dict[str, QCThresholds] = {
    "hrrr": QCThresholds(t_C=8.0, td_C=10.0, wind_ms=12.0, mslp_hPa=6.0),
    "rap": QCThresholds(t_C=10.0, td_C=12.0, wind_ms=15.0, mslp_hPa=8.0),
    "nam": QCThresholds(t_C=12.0, td_C=14.0, wind_ms=15.0, mslp_hPa=8.0),
}


# ---------------------------------------------------------------------------
# Output / plotting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputConfig:
    """Rendering defaults."""

    dpi: int = 150
    figure_size: Tuple[float, float] = (16.0, 12.0)


# ---------------------------------------------------------------------------
# Convenience singleton instances
# ---------------------------------------------------------------------------

DOMAIN = DomainConfig()
OBS = ObsConfig()
BARNES = BarnesConfig()
OUTPUT = OutputConfig()
