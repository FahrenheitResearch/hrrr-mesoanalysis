"""
Fetch surface observations from NDBC (National Data Buoy Center) marine buoys
and C-MAN coastal stations.

Primary data source is the NDBC ``latest_obs.txt`` file, which contains the
most recent observation from every active station in a single download.  If
that fails, the module falls back to fetching individual station files from
the ``realtime2`` directory using a thread pool.

No API key is required.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from mesoanalysis.obs.models import ObsCollection, SurfaceObs

log = logging.getLogger(__name__)

# -- URLs -------------------------------------------------------------------
_LATEST_OBS_URL = "https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt"
_STATION_TABLE_URL = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
_REALTIME2_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"

# -- CONUS bounding box (includes Gulf, Atlantic, Pacific, Great Lakes) -----
_LAT_MIN, _LAT_MAX = 20.0, 55.0
_LON_MIN, _LON_MAX = -130.0, -60.0

# Required columns in the latest_obs / realtime2 data
_REQUIRED_FIELDS = ("ATMP", "DEWP", "WSPD", "WDIR", "PRES")


# ---------------------------------------------------------------------------
# Wind conversion
# ---------------------------------------------------------------------------

def _wind_components(speed_ms: float, direction_deg: float) -> tuple[float, float]:
    """Convert wind speed (m/s) and meteorological direction (degrees) to u, v (m/s).

    Meteorological convention: direction is where the wind comes *from*.
    """
    dir_rad = np.radians(direction_deg)
    u = -speed_ms * np.sin(dir_rad)
    v = -speed_ms * np.cos(dir_rad)
    return float(u), float(v)


# ---------------------------------------------------------------------------
# Parsing latest_obs.txt
# ---------------------------------------------------------------------------

def _parse_latest_obs(
    text: str,
    analysis_time: datetime,
    window: timedelta,
) -> list[SurfaceObs]:
    """Parse the NDBC latest_obs.txt bulk file.

    The file is space-delimited.  First two lines are headers:
        #STN     LAT      LON  YYYY MM DD hh mm  WDIR WSPD GST  WVHT  DPD  APD  MWD  PRES  ATMP  WTMP  DEWP  VIS  PTDY  TIDE
        #text    deg      deg   yr  mo dy hr mn  degT  m/s  m/s   m   sec  sec degT   hPa  degC  degC  degC  nmi   hPa    ft

    Missing values are ``MM``.
    """
    obs_list: list[SurfaceObs] = []
    lines = text.strip().splitlines()

    # Identify header to get column positions
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and ("STN" in stripped or "LAT" in stripped):
            header_line = stripped
            data_start = i + 1
            continue
        if stripped.startswith("#"):
            data_start = i + 1
            continue

    if header_line is None:
        # Try treating first line as header
        header_line = lines[0].strip()
        data_start = 2  # skip both header lines

    # Parse header columns
    cols = header_line.lstrip("#").split()

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < len(cols):
            continue

        row = dict(zip(cols, parts))

        try:
            # Check required fields
            for f in _REQUIRED_FIELDS:
                if row.get(f, "MM") == "MM":
                    raise ValueError(f"Missing {f}")

            lat = float(row["LAT"])
            lon = float(row["LON"])

            # Filter to CONUS region
            if not (_LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX):
                continue

            # Parse observation time
            obs_time = datetime(
                int(row["YYYY"]), int(row["MM"]), int(row["DD"]),
                int(row["hh"]), int(row["mm"]),
                tzinfo=timezone.utc,
            )

            # Check time window
            if abs((obs_time - analysis_time).total_seconds()) > window.total_seconds():
                continue

            atmp = float(row["ATMP"])
            dewp = float(row["DEWP"])
            wspd = float(row["WSPD"])
            wdir = float(row["WDIR"])
            pres = float(row["PRES"])

            u, v = _wind_components(wspd, wdir)

            obs_list.append(
                SurfaceObs(
                    station=row["STN"],
                    lat=lat,
                    lon=lon,
                    time=obs_time,
                    t_C=round(atmp, 2),
                    td_C=round(dewp, 2),
                    u_ms=round(u, 2),
                    v_ms=round(v, 2),
                    mslp_hPa=round(pres, 2),
                )
            )
        except (ValueError, KeyError, TypeError):
            continue

    return obs_list


# ---------------------------------------------------------------------------
# Fallback: station table + individual realtime2 files
# ---------------------------------------------------------------------------

def _fetch_station_ids(timeout: int) -> list[tuple[str, float, float]]:
    """Fetch active station IDs and coordinates from the NDBC station table.

    Returns list of (station_id, lat, lon) tuples within the CONUS bounding box.
    """
    resp = requests.get(_STATION_TABLE_URL, timeout=timeout)
    resp.raise_for_status()

    stations: list[tuple[str, float, float]] = []
    for line in resp.text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Pipe-delimited: station_id | ... various fields with lat/lon embedded
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        station_id = parts[0].strip()
        if not station_id or station_id.upper() == "STATION":
            continue

        # The station table has lat and lon in specific columns.
        # Try to find numeric lat/lon — they appear as e.g. "38.457 N" "76.411 W"
        try:
            lat_str = parts[1].strip()
            lon_str = parts[2].strip()

            # Handle "38.457 N" / "38.457 S" format
            lat_parts = lat_str.split()
            lat = float(lat_parts[0])
            if len(lat_parts) > 1 and lat_parts[1].upper() == "S":
                lat = -lat

            lon_parts = lon_str.split()
            lon = float(lon_parts[0])
            if len(lon_parts) > 1 and lon_parts[1].upper() == "W":
                lon = -lon
            elif len(lon_parts) > 1 and lon_parts[1].upper() == "E":
                pass  # already positive

            if _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX:
                stations.append((station_id, lat, lon))
        except (ValueError, IndexError):
            continue

    log.info("Found %d CONUS-region NDBC stations from station table", len(stations))
    return stations


def _parse_realtime2(
    text: str,
    station_id: str,
    lat: float,
    lon: float,
    analysis_time: datetime,
    window: timedelta,
) -> list[SurfaceObs]:
    """Parse a single station's realtime2 data file.

    Format (space-delimited):
        #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY TIDE
        #yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa   ft
    """
    obs_list: list[SurfaceObs] = []
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return obs_list

    # First line is header with column names
    header = lines[0].lstrip("#").split()
    # Second line is units — skip it
    # Map column name to index position
    col_name_to_key = {
        "YY": "YY", "#YY": "YY",
        "YYYY": "YY",
    }

    for line in lines[2:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < len(header):
            continue

        row = dict(zip(header, parts))
        # Normalize year column
        year_key = "YY" if "YY" in row else "YYYY" if "YYYY" in row else None
        if year_key is None:
            continue

        try:
            for f in _REQUIRED_FIELDS:
                if row.get(f, "MM") == "MM":
                    raise ValueError(f"Missing {f}")

            year = int(row[year_key])
            if year < 100:
                year += 2000

            obs_time = datetime(
                year, int(row["MM"]), int(row["DD"]),
                int(row["hh"]), int(row["mm"]),
                tzinfo=timezone.utc,
            )

            if abs((obs_time - analysis_time).total_seconds()) > window.total_seconds():
                continue

            atmp = float(row["ATMP"])
            dewp = float(row["DEWP"])
            wspd = float(row["WSPD"])
            wdir = float(row["WDIR"])
            pres = float(row["PRES"])

            u, v = _wind_components(wspd, wdir)

            obs_list.append(
                SurfaceObs(
                    station=station_id,
                    lat=lat,
                    lon=lon,
                    time=obs_time,
                    t_C=round(atmp, 2),
                    td_C=round(dewp, 2),
                    u_ms=round(u, 2),
                    v_ms=round(v, 2),
                    mslp_hPa=round(pres, 2),
                )
            )
        except (ValueError, KeyError, TypeError):
            continue

    return obs_list


def _fetch_realtime2_fallback(
    analysis_time: datetime,
    window: timedelta,
    timeout: int,
) -> list[SurfaceObs]:
    """Fallback: fetch station table, then individual realtime2 files via thread pool."""
    try:
        stations = _fetch_station_ids(timeout)
    except requests.RequestException as exc:
        log.error("Failed to fetch NDBC station table: %s", exc)
        return []

    all_obs: list[SurfaceObs] = []

    def _fetch_one(station_id: str, lat: float, lon: float) -> list[SurfaceObs]:
        url = _REALTIME2_URL.format(station_id=station_id)
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return _parse_realtime2(resp.text, station_id, lat, lon, analysis_time, window)
        except requests.RequestException:
            return []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_fetch_one, sid, lat, lon): sid
            for sid, lat, lon in stations
        }
        for future in as_completed(futures):
            try:
                all_obs.extend(future.result())
            except Exception as exc:
                log.warning("NDBC fetch error for %s: %s", futures[future], exc)

    return all_obs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ndbc_obs(
    analysis_time: datetime,
    window_minutes: int = 60,
    timeout: int = 30,
) -> ObsCollection:
    """Fetch NDBC buoy and C-MAN station observations near *analysis_time*.

    Parameters
    ----------
    analysis_time : datetime
        Centre of the observation window (should be timezone-aware UTC).
    window_minutes : int
        Half-width of the time window in minutes (default +-60 min).
        Buoys typically report hourly, so a wider window is appropriate.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    ObsCollection
    """
    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)

    window = timedelta(minutes=window_minutes)
    all_obs: list[SurfaceObs] = []

    # -- Primary: bulk latest_obs.txt ----------------------------------------
    try:
        log.info("Fetching NDBC latest_obs.txt ...")
        resp = requests.get(_LATEST_OBS_URL, timeout=timeout)
        resp.raise_for_status()
        all_obs = _parse_latest_obs(resp.text, analysis_time, window)
        log.info("Parsed %d obs from latest_obs.txt", len(all_obs))
    except requests.RequestException as exc:
        log.warning("Failed to fetch latest_obs.txt (%s), falling back to realtime2", exc)
        all_obs = _fetch_realtime2_fallback(analysis_time, window, timeout)

    # -- De-duplicate: keep obs closest to analysis_time per station ----------
    best: dict[str, SurfaceObs] = {}
    for o in all_obs:
        dt = abs((o.time - analysis_time).total_seconds())
        prev = best.get(o.station)
        if prev is None or dt < abs((prev.time - analysis_time).total_seconds()):
            best[o.station] = o

    result = ObsCollection(obs=list(best.values()))
    log.info(
        "NDBC: %d unique stations (%d raw obs) for %s",
        len(result),
        len(all_obs),
        analysis_time.strftime("%Y-%m-%d %H:%MZ"),
    )
    return result
