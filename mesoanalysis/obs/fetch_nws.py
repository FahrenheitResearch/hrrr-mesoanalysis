"""
Fetch surface observations from the NWS API (api.weather.gov).

No API key required -- only a ``User-Agent`` header.  This module
complements the IEM ASOS fetcher in ``fetch.py`` by pulling from the
official NWS observation endpoints, which may include stations not
present in the IEM ASOS networks.

Because the NWS API is rate-limited, we:
* limit concurrency to 5 threads,
* include an identifying User-Agent header,
* fetch station lists per-state first, then retrieve observations.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

from mesoanalysis.obs.models import ObsCollection, SurfaceObs

log = logging.getLogger(__name__)

_USER_AGENT = "hrrr-mesoanalysis/1.0 (surface obs fetcher)"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}

_NWS_BASE = "https://api.weather.gov"

# CONUS state two-letter codes (same list used in fetch.py)
CONUS_STATES: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
]


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _extract_value(obj) -> Optional[float]:
    """Extract a numeric value from an NWS quantity object.

    NWS returns values as ``{"value": 23.5, "unitCode": "wmoUnit:degC"}``
    or ``{"value": null, ...}``.  Returns *None* when the value is missing.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        v = obj.get("value")
        return float(v) if v is not None else None
    # Fallback: bare numeric
    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


def _wind_speed_to_ms(obj) -> Optional[float]:
    """Convert an NWS wind-speed quantity to m/s.

    Handles ``wmoUnit:m_s-1`` (already m/s) and ``wmoUnit:km_h-1``
    (divide by 3.6).
    """
    if obj is None:
        return None
    val = _extract_value(obj)
    if val is None:
        return None
    unit = obj.get("unitCode", "") if isinstance(obj, dict) else ""
    if "km_h" in unit or "km/h" in unit:
        return val / 3.6
    # Default: assume m/s
    return val


def _pressure_to_hpa(obj) -> Optional[float]:
    """Convert an NWS pressure quantity to hPa.

    Handles ``wmoUnit:Pa`` (divide by 100) and values already in hPa.
    """
    if obj is None:
        return None
    val = _extract_value(obj)
    if val is None:
        return None
    unit = obj.get("unitCode", "") if isinstance(obj, dict) else ""
    if "Pa" in unit and "hPa" not in unit:
        # Raw pascals → hectopascals
        return val / 100.0
    return val


def _wind_components(speed_ms: float, direction_deg: float) -> tuple[float, float]:
    """Convert wind speed (m/s) and meteorological direction (degrees) to u, v.

    Meteorological convention: direction is where the wind comes *from*.
    u = -speed * sin(dir),  v = -speed * cos(dir).
    """
    dir_rad = np.radians(direction_deg)
    u = -speed_ms * np.sin(dir_rad)
    v = -speed_ms * np.cos(dir_rad)
    return float(u), float(v)


# ---------------------------------------------------------------------------
# Station list
# ---------------------------------------------------------------------------

def _fetch_station_ids(
    state: str,
    session: requests.Session,
    timeout: int,
) -> list[tuple[str, float, float]]:
    """Return list of (station_id, lat, lon) for a state.

    Uses ``GET /stations?state={ST}&limit=500``.
    """
    url = f"{_NWS_BASE}/stations?state={state}&limit=500"
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("Failed to fetch station list for %s: %s", state, exc)
        return []

    stations: list[tuple[str, float, float]] = []

    # Prefer "features" array (GeoJSON)
    features = data.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        sid = props.get("stationIdentifier")
        if not sid:
            continue
        # Coordinates come from geometry (GeoJSON: [lon, lat])
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        if coords[0] is not None and coords[1] is not None:
            stations.append((sid, float(coords[1]), float(coords[0])))

    if not stations:
        # Fallback: "observationStations" is a list of URLs
        for url_str in data.get("observationStations", []):
            sid = url_str.rsplit("/", 1)[-1]
            # No coordinates available from this format; skip
            # (we'll get them from the observation response later)
            stations.append((sid, 0.0, 0.0))

    log.debug("State %s: %d stations", state, len(stations))
    return stations


# ---------------------------------------------------------------------------
# Single-station observation fetch
# ---------------------------------------------------------------------------

def _fetch_latest_obs(
    station_id: str,
    lat: float,
    lon: float,
    analysis_time: datetime,
    window: timedelta,
    session: requests.Session,
    timeout: int,
) -> Optional[SurfaceObs]:
    """Fetch the latest observation for a single station and parse it.

    Returns *None* if the request fails or essential fields are missing.
    """
    url = f"{_NWS_BASE}/stations/{station_id}/observations/latest"
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None  # station has no recent obs
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.debug("Obs fetch failed for %s: %s", station_id, exc)
        return None

    props = data.get("properties", {})

    # -- Observation timestamp ------------------------------------------------
    timestamp_str = props.get("timestamp")
    if not timestamp_str:
        return None
    try:
        obs_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    # Check time window
    if abs((obs_time - analysis_time).total_seconds()) > window.total_seconds():
        return None

    # -- Extract values -------------------------------------------------------
    t_c = _extract_value(props.get("temperature"))
    td_c = _extract_value(props.get("dewpoint"))
    wdir = _extract_value(props.get("windDirection"))
    wspd = _wind_speed_to_ms(props.get("windSpeed"))

    # Pressure: prefer sea-level pressure, fall back to barometric
    mslp = _pressure_to_hpa(props.get("seaLevelPressure"))
    if mslp is None:
        mslp = _pressure_to_hpa(props.get("barometricPressure"))

    # All essential fields must be present
    if any(v is None for v in (t_c, td_c, wdir, wspd, mslp)):
        return None

    # Sanity bounds
    if not (-80.0 <= t_c <= 60.0):
        return None
    if not (800.0 <= mslp <= 1100.0):
        return None

    u, v = _wind_components(wspd, wdir)

    # Use coordinates from station list; update from obs if available
    obs_lat = lat
    obs_lon = lon
    # Some responses include geometry at top level
    geom = data.get("geometry", {})
    coords = geom.get("coordinates", [])
    if len(coords) >= 2 and coords[0] is not None:
        obs_lon, obs_lat = float(coords[0]), float(coords[1])
    # If we still have placeholder coords, skip
    if obs_lat == 0.0 and obs_lon == 0.0:
        return None

    return SurfaceObs(
        station=station_id,
        lat=obs_lat,
        lon=obs_lon,
        time=obs_time,
        t_C=round(t_c, 2),
        td_C=round(td_c, 2),
        u_ms=round(u, 2),
        v_ms=round(v, 2),
        mslp_hPa=round(mslp, 2),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_nws_obs(
    analysis_time: datetime,
    window_minutes: int = 20,
    timeout: int = 30,
) -> ObsCollection:
    """Fetch surface observations from the NWS API.

    Parameters
    ----------
    analysis_time : datetime
        Centre of the observation window (should be timezone-aware UTC).
    window_minutes : int
        Half-width of the time window in minutes (default +-20 min).
    timeout : int
        HTTP request timeout in seconds for each individual request.

    Returns
    -------
    ObsCollection
        Deduplicated observations, one per station.
    """
    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)

    window = timedelta(minutes=window_minutes)
    session = requests.Session()
    session.headers.update(_HEADERS)

    # -- Phase 1: collect station lists for all CONUS states ------------------
    log.info("NWS API: fetching station lists for %d states ...", len(CONUS_STATES))
    all_stations: list[tuple[str, float, float]] = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_fetch_station_ids, st, session, timeout): st
            for st in CONUS_STATES
        }
        for future in as_completed(futures):
            try:
                all_stations.extend(future.result())
            except Exception as exc:
                log.warning(
                    "Station-list thread error for %s: %s",
                    futures[future], exc,
                )

    # Deduplicate by station ID (same station may appear in multiple states)
    seen: dict[str, tuple[float, float]] = {}
    for sid, lat, lon in all_stations:
        if sid not in seen:
            seen[sid] = (lat, lon)
    unique_stations = [(sid, lat, lon) for sid, (lat, lon) in seen.items()]
    log.info("NWS API: %d unique stations across all states", len(unique_stations))

    # -- Phase 2: fetch latest observation per station ------------------------
    obs_list: list[SurfaceObs] = []

    def _fetch_one(item: tuple[str, float, float]) -> Optional[SurfaceObs]:
        sid, lat, lon = item
        return _fetch_latest_obs(sid, lat, lon, analysis_time, window, session, timeout)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_one, s): s[0] for s in unique_stations}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 200 == 0:
                log.info("NWS API: %d / %d stations fetched", done, len(unique_stations))
            try:
                obs = future.result()
                if obs is not None:
                    obs_list.append(obs)
            except Exception as exc:
                log.debug("Obs thread error for %s: %s", futures[future], exc)

    result = ObsCollection(obs=obs_list)
    log.info(
        "NWS API: %d observations retrieved for %s",
        len(result),
        analysis_time.strftime("%Y-%m-%d %H:%MZ"),
    )
    return result
