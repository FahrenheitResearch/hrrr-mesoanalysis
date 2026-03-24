"""
Fetch surface observations from IEM (Iowa Environmental Mesonet) networks
beyond basic ASOS stations: state mesonets, RWIS, AWOS, and COOP networks.

Uses the IEM ``currents.json`` API which returns the most recent observation
for every station in a given network.  No API key is required.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from mesoanalysis.obs.models import ObsCollection, SurfaceObs

log = logging.getLogger(__name__)

# -- CONUS state two-letter codes --------------------------------------------
_CONUS_STATES: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
]

# -- Network IDs to fetch ---------------------------------------------------
# Per-state networks (expanded for every CONUS state):
# Note: AWOS is included under ASOS in IEM, so we skip it.
# COOP stations rarely have real-time hourly data but included for completeness.
# DCP (Data Collection Platform) includes RAWS, mesonets, ALERT, etc. via GOES.
_PER_STATE_SUFFIXES: list[str] = ["COOP", "RWIS", "DCP"]

# No specific mesonet network IDs — IEM does not host state mesonets like
# OK_MESONET under the currents.json API.  Those stations appear as DCP or
# are accessible via Synoptic/MesoWest (requires API key).
_SPECIFIC_NETWORKS: list[str] = []

# IEM currents API endpoint
_IEM_CURRENTS_URL = "https://mesonet.agron.iastate.edu/api/1/currents.json"


def _build_network_list() -> list[str]:
    """Build the full list of IEM network IDs to query."""
    networks: list[str] = []
    for st in _CONUS_STATES:
        for suffix in _PER_STATE_SUFFIXES:
            networks.append(f"{st}_{suffix}")
    networks.extend(_SPECIFIC_NETWORKS)
    return networks


NETWORKS: list[str] = _build_network_list()


# ---------------------------------------------------------------------------
# Unit-conversion helpers
# ---------------------------------------------------------------------------

def _f_to_c(f: float) -> float:
    """Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0


def _wind_components(speed_kts: float, direction_deg: float) -> tuple[float, float]:
    """Convert wind speed (knots) and direction (degrees) to u, v (m/s).

    Meteorological convention: direction is where the wind comes *from*.
    u = -speed * sin(dir),  v = -speed * cos(dir).
    """
    spd_ms = speed_kts * 0.514444
    dir_rad = np.radians(direction_deg)
    u = -spd_ms * np.sin(dir_rad)
    v = -spd_ms * np.cos(dir_rad)
    return float(u), float(v)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    """Return True if the value represents a missing observation."""
    if value is None:
        return True
    s = str(value).strip()
    return s in ("M", "", "null", "None")


def _parse_currents(data: list[dict], analysis_time: datetime,
                    window_minutes: int) -> list[SurfaceObs]:
    """Parse the ``data`` array from an IEM currents.json response."""
    obs_list: list[SurfaceObs] = []
    cutoff_start = analysis_time - timedelta(minutes=window_minutes)
    cutoff_end = analysis_time + timedelta(minutes=window_minutes)

    for entry in data:
        try:
            # Temperature, dewpoint, wind are required; pressure is optional
            for fld in ("tmpf", "dwpf", "drct", "sknt"):
                if _is_missing(entry.get(fld)):
                    raise ValueError(f"Missing {fld}")

            tmpf = float(entry["tmpf"])
            dwpf = float(entry["dwpf"])
            drct = float(entry["drct"])
            sknt = float(entry["sknt"])

            # Pressure: try mslp, then alti (inHg -> hPa), else NaN
            if not _is_missing(entry.get("mslp")):
                mslp = float(entry["mslp"])
            elif not _is_missing(entry.get("alti")):
                mslp = float(entry["alti"]) * 33.8639
            else:
                mslp = float("nan")

            lat = float(entry["lat"])
            lon = float(entry["lon"])

            # Parse observation time
            time_str = entry.get("utc_valid", "")
            if _is_missing(time_str):
                continue
            # IEM returns ISO-ish timestamps like "2024-06-15T18:53:00Z"
            time_str = str(time_str).replace("Z", "+00:00")
            obs_time = datetime.fromisoformat(time_str)
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)

            # Filter to time window
            if obs_time < cutoff_start or obs_time > cutoff_end:
                continue

            t_c = _f_to_c(tmpf)
            td_c = _f_to_c(dwpf)
            u, v = _wind_components(sknt, drct)

            station = str(entry.get("station", "")).strip()
            if not station:
                continue

            obs_list.append(
                SurfaceObs(
                    station=station,
                    lat=lat,
                    lon=lon,
                    time=obs_time,
                    t_C=round(t_c, 2),
                    td_C=round(td_c, 2),
                    u_ms=round(u, 2),
                    v_ms=round(v, 2),
                    mslp_hPa=round(mslp, 2),
                )
            )
        except (ValueError, KeyError, TypeError):
            continue

    return obs_list


# ---------------------------------------------------------------------------
# Network fetching
# ---------------------------------------------------------------------------

def _fetch_network(network: str, analysis_time: datetime,
                   window_minutes: int, timeout: int) -> list[SurfaceObs]:
    """Fetch and parse current observations for a single IEM network."""
    url = f"{_IEM_CURRENTS_URL}?network={network}"
    try:
        log.debug("Fetching %s ...", network)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        log.warning("Failed to fetch %s: %s", network, exc)
        return []
    except ValueError:
        log.warning("Invalid JSON response from %s", network)
        return []

    data = payload.get("data", [])
    if not data:
        return []

    return _parse_currents(data, analysis_time, window_minutes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_mesonet_obs(
    analysis_time: datetime,
    window_minutes: int = 20,
    timeout: int = 30,
) -> ObsCollection:
    """Fetch mesonet / RWIS / AWOS / COOP observations from IEM.

    Queries the IEM ``currents.json`` API for all configured networks
    concurrently and returns a deduplicated :class:`ObsCollection`.

    Parameters
    ----------
    analysis_time : datetime
        Centre of the observation window (should be timezone-aware UTC).
    window_minutes : int
        Half-width of the time window in minutes (default +-20 min).
    timeout : int
        HTTP request timeout in seconds per network query.

    Returns
    -------
    ObsCollection
    """
    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)

    all_obs: list[SurfaceObs] = []

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(_fetch_network, net, analysis_time, window_minutes, timeout): net
            for net in NETWORKS
        }
        for future in as_completed(futures):
            net = futures[future]
            try:
                all_obs.extend(future.result())
            except Exception as exc:
                log.warning("Fetch thread error for %s: %s", net, exc)

    # Deduplicate by station ID, keeping the most recent observation
    best: dict[str, SurfaceObs] = {}
    for o in all_obs:
        prev = best.get(o.station)
        if prev is None or o.time > prev.time:
            best[o.station] = o

    result = ObsCollection(obs=list(best.values()))
    log.info(
        "Mesonet fetch: %d unique stations (%d raw obs) for %s",
        len(result),
        len(all_obs),
        analysis_time.strftime("%Y-%m-%d %H:%MZ"),
    )
    return result
