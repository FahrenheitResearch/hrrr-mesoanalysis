"""Central configuration for the mesoscale analysis system."""

from dataclasses import dataclass, field
from typing import List, Tuple

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

    kappa: float = 5.052e9
    """First-pass smoothing parameter (m^2).  Tune to station spacing."""

    gamma: float = 0.3
    """Convergence parameter applied on second pass (kappa2 = gamma * kappa)."""

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
