"""
Data models for surface observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class SurfaceObs:
    """Single surface (METAR/ASOS) observation."""

    station: str
    lat: float
    lon: float
    time: datetime
    t_C: float          # temperature, Celsius
    td_C: float         # dewpoint, Celsius
    u_ms: float         # u-wind component, m/s
    v_ms: float         # v-wind component, m/s
    mslp_hPa: float     # mean sea-level pressure, hPa

    @property
    def wind_speed_ms(self) -> float:
        """Scalar wind speed (m/s)."""
        return float(np.hypot(self.u_ms, self.v_ms))

    @property
    def wind_dir_deg(self) -> float:
        """Meteorological wind direction (degrees, 0-360)."""
        return float((270.0 - np.degrees(np.arctan2(self.v_ms, self.u_ms))) % 360.0)


@dataclass
class ObsCollection:
    """Container for a set of surface observations with convenient accessors."""

    obs: list[SurfaceObs] = field(default_factory=list)

    # -- size helpers --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.obs)

    def __iter__(self):
        return iter(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx]

    # -- coordinate arrays ---------------------------------------------------
    @property
    def lats(self) -> np.ndarray:
        return np.array([o.lat for o in self.obs])

    @property
    def lons(self) -> np.ndarray:
        return np.array([o.lon for o in self.obs])

    # -- scalar field arrays -------------------------------------------------
    @property
    def t_array(self) -> np.ndarray:
        """Temperature (C) for every obs."""
        return np.array([o.t_C for o in self.obs])

    @property
    def td_array(self) -> np.ndarray:
        """Dewpoint (C) for every obs."""
        return np.array([o.td_C for o in self.obs])

    @property
    def u_array(self) -> np.ndarray:
        """u-wind (m/s) for every obs."""
        return np.array([o.u_ms for o in self.obs])

    @property
    def v_array(self) -> np.ndarray:
        """v-wind (m/s) for every obs."""
        return np.array([o.v_ms for o in self.obs])

    @property
    def mslp_array(self) -> np.ndarray:
        """Mean sea-level pressure (hPa) for every obs."""
        return np.array([o.mslp_hPa for o in self.obs])

    @property
    def wind_speed_array(self) -> np.ndarray:
        return np.hypot(self.u_array, self.v_array)

    @property
    def stations(self) -> list[str]:
        return [o.station for o in self.obs]

    # -- filtering -----------------------------------------------------------
    def filter(self, mask: np.ndarray) -> "ObsCollection":
        """Return a new collection keeping only obs where *mask* is True."""
        return ObsCollection(obs=[o for o, keep in zip(self.obs, mask) if keep])
