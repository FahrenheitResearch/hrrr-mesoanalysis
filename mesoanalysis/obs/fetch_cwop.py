"""
Fetch surface observations from CWOP/APRS citizen weather stations via FindU.

CWOP (Citizen Weather Observer Program) stations report via APRS and are
aggregated by FindU (findu.com).  This module queries the ``wxnear.cgi``
endpoint on a geographic grid covering CONUS, parses the returned HTML tables,
and produces an :class:`ObsCollection`.

No API key is required.  Data quality is lower than ASOS/METAR -- the QC
module is expected to handle outlier filtering downstream.
"""

from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

from mesoanalysis.obs.models import ObsCollection, SurfaceObs

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WXNEAR_URL = "http://www.findu.com/cgi-bin/wxnear.cgi"

# CONUS bounding box (degrees)
_LAT_MIN, _LAT_MAX = 25.0, 49.0
_LON_MIN, _LON_MAX = -124.0, -67.0

# Grid spacing in degrees (~200-250 miles at mid-latitudes)
_GRID_STEP = 3.5

# Search radius per grid point (miles) -- overlap ensures coverage
_MAX_DIST_MI = 200

# Maximum report age to request from FindU (hours)
_MAX_AGE_HR = 2


# ---------------------------------------------------------------------------
# Unit / meteorology helpers
# ---------------------------------------------------------------------------

def _f_to_c(f: float) -> float:
    """Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0


def _mph_to_ms(mph: float) -> float:
    """Miles per hour to metres per second."""
    return mph * 0.44704


def _knots_to_ms(kts: float) -> float:
    return kts * 0.514444


def _dewpoint_c(t_c: float, rh: float) -> float:
    """Compute dewpoint (C) from temperature (C) and relative humidity (%).

    Uses the Magnus formula with Alduchov (1996) coefficients.
    """
    if rh <= 0.0:
        rh = 0.1  # guard against log(0)
    if rh > 100.0:
        rh = 100.0
    a = 17.625
    b = 243.04
    gamma = math.log(rh / 100.0) + (a * t_c) / (b + t_c)
    td = (b * gamma) / (a - gamma)
    return td


def _wind_components(speed_ms: float, direction_deg: float) -> tuple[float, float]:
    """Convert scalar wind speed (m/s) and met direction (deg) to u, v (m/s).

    Meteorological convention: direction is where the wind comes *from*.
    """
    dir_rad = np.radians(direction_deg)
    u = -speed_ms * float(np.sin(dir_rad))
    v = -speed_ms * float(np.cos(dir_rad))
    return u, v


_DIRECTION_MAP: dict[str, float] = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def _parse_wind_direction(text: str) -> Optional[float]:
    """Parse a wind direction string (compass or numeric degrees)."""
    text = text.strip().upper()
    if text in _DIRECTION_MAP:
        return _DIRECTION_MAP[text]
    try:
        deg = float(text)
        if 0.0 <= deg <= 360.0:
            return deg
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _conus_grid() -> list[tuple[float, float]]:
    """Return a list of (lat, lon) grid points spanning CONUS."""
    points: list[tuple[float, float]] = []
    lat = _LAT_MIN
    while lat <= _LAT_MAX:
        lon = _LON_MIN
        while lon <= _LON_MAX:
            points.append((round(lat, 1), round(lon, 1)))
            lon += _GRID_STEP
        lat += _GRID_STEP
    log.debug("CWOP grid: %d query points", len(points))
    return points


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

# FindU wxnear.cgi returns an HTML page with a <table> of station rows.
# Each row typically contains:
#   - A link with the station callsign
#   - Latitude / longitude text
#   - Weather data: temperature (F), humidity (%), pressure (mb), wind
#
# The exact format can vary, so we use multiple regex strategies.

# Pattern for APRS-style weather data embedded in page text.
# APRS weather format:  cDDDsSSS...tTTT...hHH...bBBBBB
#   c = wind direction (degrees, 3 digits)
#   s = sustained wind speed (mph, 3 digits)
#   g = gust speed (mph, 3 digits)  [optional]
#   t = temperature (F, 3 digits, can be negative like t-05)
#   h = humidity (%, 2 digits, 00 = 100%)
#   b = barometric pressure (tenths of hPa, 5 digits, e.g. 10132 = 1013.2 hPa)
_APRS_WX_RE = re.compile(
    r"c(\d{3})"         # wind direction (degrees)
    r"s(\d{3})"         # wind speed (mph)
    r"(?:g(\d{3}))?"    # gust (mph) [optional]
    r".*?"
    r"t(-?\d{2,3})"     # temperature (F)
    r".*?"
    r"(?:h(\d{2}))?"    # humidity (%) [optional, 00=100]
    r".*?"
    r"(?:b(\d{5}))?"    # pressure (tenths hPa) [optional]
)

# Regex to extract station callsign from a FindU link
_CALLSIGN_RE = re.compile(
    r'<a\s+href=["\'][^"\']*[?&]call=([A-Za-z0-9-]+)["\']',
    re.IGNORECASE,
)

# Regex for lat/lon in decimal degrees (common FindU format)
# e.g. "40.1234" and "-90.5678" or "40.1234N" "90.5678W"
_LATLON_DECIMAL_RE = re.compile(
    r"(-?\d{1,3}\.\d+)\s*([NS])?\s*[,/\s]+\s*(-?\d{1,3}\.\d+)\s*([EW])?"
)

# Human-readable weather line:
# "72°F  45%  1013.2mb  S at 5mph  gusting to 8"
_HUMAN_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°|&deg;)\s*F", re.IGNORECASE)
_HUMAN_HUMIDITY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_HUMAN_PRESSURE_RE = re.compile(r"(\d{3,4}(?:\.\d+)?)\s*(?:mb|hPa|mbar)", re.IGNORECASE)
_HUMAN_WIND_RE = re.compile(
    r"(N|NNE|NE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW|\d{1,3})"
    r"\s+at\s+(\d+(?:\.\d+)?)\s*(?:mph|kt|kts|knots)",
    re.IGNORECASE,
)


def _parse_aprs_wx(text: str) -> Optional[dict]:
    """Try to extract weather data from an APRS weather string."""
    m = _APRS_WX_RE.search(text)
    if not m:
        return None

    wdir = int(m.group(1))
    wspd_mph = int(m.group(2))
    temp_f = int(m.group(3)) if m.group(3) is not None else int(m.group(4))
    temp_f = int(m.group(4))

    humidity = None
    if m.group(5) is not None:
        h = int(m.group(5))
        humidity = 100.0 if h == 0 else float(h)

    pressure = None
    if m.group(6) is not None:
        pressure = int(m.group(6)) / 10.0  # tenths of hPa -> hPa

    if humidity is None or pressure is None:
        return None

    t_c = _f_to_c(temp_f)
    td_c = _dewpoint_c(t_c, humidity)
    spd_ms = _mph_to_ms(wspd_mph)
    u, v = _wind_components(spd_ms, float(wdir))

    return {
        "t_C": round(t_c, 2),
        "td_C": round(td_c, 2),
        "u_ms": round(u, 2),
        "v_ms": round(v, 2),
        "mslp_hPa": round(pressure, 2),
    }


def _parse_human_wx(text: str) -> Optional[dict]:
    """Try to extract weather data from human-readable weather text."""
    temp_m = _HUMAN_TEMP_RE.search(text)
    hum_m = _HUMAN_HUMIDITY_RE.search(text)
    pres_m = _HUMAN_PRESSURE_RE.search(text)
    wind_m = _HUMAN_WIND_RE.search(text)

    if not (temp_m and hum_m and pres_m and wind_m):
        return None

    temp_f = float(temp_m.group(1))
    humidity = float(hum_m.group(1))
    pressure = float(pres_m.group(1))
    wind_dir_str = wind_m.group(1)
    wind_spd_mph = float(wind_m.group(2))

    # Sanity-check pressure range (allow station pressure and MSLP)
    if pressure < 800.0 or pressure > 1100.0:
        return None

    wdir = _parse_wind_direction(wind_dir_str)
    if wdir is None:
        return None

    t_c = _f_to_c(temp_f)
    td_c = _dewpoint_c(t_c, humidity)

    # Determine speed units from the match
    unit = wind_m.group(0).lower()
    if "kt" in unit or "knot" in unit:
        spd_ms = _knots_to_ms(wind_spd_mph)
    else:
        spd_ms = _mph_to_ms(wind_spd_mph)

    u, v = _wind_components(spd_ms, wdir)

    return {
        "t_C": round(t_c, 2),
        "td_C": round(td_c, 2),
        "u_ms": round(u, 2),
        "v_ms": round(v, 2),
        "mslp_hPa": round(pressure, 2),
    }


def _parse_wxnear_html(html: str) -> list[dict]:
    """Parse the wxnear.cgi HTML response into a list of raw station dicts.

    Each dict has keys: callsign, lat, lon, and weather fields.
    Returns only stations where all required fields were parsed.
    """
    results: list[dict] = []

    # Strategy 1: Split by table rows and parse each one
    # FindU uses <tr> rows; split on them.
    rows = re.split(r"<tr[^>]*>", html, flags=re.IGNORECASE)

    for row in rows:
        # Strip HTML tags for easier text parsing, but keep raw for link extraction
        raw_row = row

        # Extract callsign
        cs_match = _CALLSIGN_RE.search(raw_row)
        if not cs_match:
            continue
        callsign = cs_match.group(1).upper()

        # Skip non-CWOP stations (typical CWOP prefixes: CW, DW, EW, FW, GW,
        # or single letter + digits).  Also accept generic APRS callsigns.
        # We are permissive here -- QC will handle bad stations.

        # Strip all HTML tags for text parsing
        text = re.sub(r"<[^>]+>", " ", raw_row)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Extract lat/lon
        ll_match = _LATLON_DECIMAL_RE.search(text)
        if not ll_match:
            continue

        lat = float(ll_match.group(1))
        if ll_match.group(2) and ll_match.group(2).upper() == "S":
            lat = -lat
        lon = float(ll_match.group(3))
        if ll_match.group(4) and ll_match.group(4).upper() == "W":
            lon = -lon
        # If lon is positive but we expect CONUS (negative), try negating
        if lon > 0 and _LON_MIN < -lon < _LON_MAX + 5:
            lon = -lon

        # Basic lat/lon sanity for CONUS (be generous for border stations)
        if not (20.0 <= lat <= 55.0 and -130.0 <= lon <= -60.0):
            continue

        # Try APRS weather format first, then human-readable
        wx = _parse_aprs_wx(text)
        if wx is None:
            wx = _parse_human_wx(text)
        if wx is None:
            # Try parsing APRS from the raw HTML (sometimes in attributes)
            wx = _parse_aprs_wx(raw_row)
        if wx is None:
            continue

        wx["callsign"] = callsign
        wx["lat"] = lat
        wx["lon"] = lon
        results.append(wx)

    # Strategy 2: If table-row parsing found nothing, try line-by-line
    # with APRS raw strings (some FindU pages embed raw APRS packets).
    if not results:
        # Look for raw APRS packets like:  CALL>...:@DDHHMMzDDMM.HHN/DDDMM.HHW_cDDDsSSStTTT...
        aprs_packet_re = re.compile(
            r"([A-Z0-9-]{3,9})>[^:]+:"
            r"[!=/@][^_]*_"
            r"(c\d{3}s\d{3}.*?(?:t-?\d{2,3}).*?)(?:\s|<|$)"
        )
        aprs_pos_re = re.compile(
            r"(\d{2})(\d{2}\.\d+)([NS])[/\\](\d{3})(\d{2}\.\d+)([EW])"
        )
        for pm in aprs_packet_re.finditer(html):
            callsign = pm.group(1).upper()
            wx_str = pm.group(2)
            wx = _parse_aprs_wx(wx_str)
            if wx is None:
                continue

            # Extract position from the same packet region
            pos_search = aprs_pos_re.search(html[max(0, pm.start() - 200):pm.end()])
            if not pos_search:
                continue

            lat_deg = int(pos_search.group(1))
            lat_min = float(pos_search.group(2))
            lat = lat_deg + lat_min / 60.0
            if pos_search.group(3) == "S":
                lat = -lat

            lon_deg = int(pos_search.group(4))
            lon_min = float(pos_search.group(5))
            lon = lon_deg + lon_min / 60.0
            if pos_search.group(6) == "W":
                lon = -lon

            if not (20.0 <= lat <= 55.0 and -130.0 <= lon <= -60.0):
                continue

            wx["callsign"] = callsign
            wx["lat"] = lat
            wx["lon"] = lon
            results.append(wx)

    return results


# ---------------------------------------------------------------------------
# Network fetch
# ---------------------------------------------------------------------------

def _fetch_grid_point(
    lat: float,
    lon: float,
    max_age_hr: int,
    timeout: int,
) -> list[dict]:
    """Fetch and parse wxnear results for a single grid point."""
    params = {
        "lat": f"{lat:.1f}",
        "lon": f"{lon:.1f}",
        "max_dist": str(_MAX_DIST_MI),
        "max_age": str(max_age_hr),
    }
    try:
        resp = requests.get(
            _WXNEAR_URL,
            params=params,
            timeout=timeout,
            verify=False,  # FindU has intermittent SSL issues
        )
        resp.raise_for_status()
        return _parse_wxnear_html(resp.text)
    except requests.RequestException as exc:
        log.debug("wxnear query failed for (%.1f, %.1f): %s", lat, lon, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_cwop_obs(
    analysis_time: datetime,
    window_minutes: int = 30,
    timeout: int = 20,
) -> ObsCollection:
    """Fetch CWOP citizen weather station observations across CONUS.

    Queries FindU's ``wxnear.cgi`` on a geographic grid covering the
    contiguous United States, parses the HTML responses, and returns an
    :class:`ObsCollection` with station IDs prefixed ``CWOP_``.

    Parameters
    ----------
    analysis_time : datetime
        Nominal analysis time (timezone-aware UTC preferred).  Used only for
        the ``time`` field stamped on returned observations -- FindU's
        ``max_age`` parameter controls the recency filter.
    window_minutes : int
        Not directly used for the FindU query (which uses ``max_age`` in
        hours), but kept for API symmetry with :func:`fetch_surface_obs`.
        The ``max_age`` sent to FindU is ``max(2, ceil(window_minutes / 30))``.
    timeout : int
        HTTP request timeout in seconds for each grid-point query.

    Returns
    -------
    ObsCollection
        De-duplicated observations from CWOP stations.  Station IDs are
        prefixed with ``CWOP_`` to distinguish them from ASOS/METAR sites.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)

    max_age = max(2, math.ceil(window_minutes / 30))
    grid = _conus_grid()

    log.info(
        "Fetching CWOP obs: %d grid points, max_age=%dh, timeout=%ds",
        len(grid), max_age, timeout,
    )

    all_raw: list[dict] = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_fetch_grid_point, lat, lon, max_age, timeout): (lat, lon)
            for lat, lon in grid
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 25 == 0:
                log.info("CWOP grid progress: %d / %d", done_count, len(futures))
            try:
                all_raw.extend(future.result())
            except Exception as exc:
                pt = futures[future]
                log.debug("CWOP thread error at (%.1f, %.1f): %s", pt[0], pt[1], exc)

    # De-duplicate by callsign -- keep the first occurrence (arbitrary but
    # consistent; all reports for a given station should carry the same data
    # within the max_age window).
    seen: dict[str, dict] = {}
    for entry in all_raw:
        cs = entry["callsign"]
        if cs not in seen:
            seen[cs] = entry

    log.info(
        "CWOP: %d raw records -> %d unique stations",
        len(all_raw), len(seen),
    )

    # Build SurfaceObs list
    obs_list: list[SurfaceObs] = []
    for cs, data in seen.items():
        try:
            obs_list.append(
                SurfaceObs(
                    station=f"CWOP_{cs}",
                    lat=data["lat"],
                    lon=data["lon"],
                    time=analysis_time,  # CWOP doesn't give per-obs timestamps easily
                    t_C=data["t_C"],
                    td_C=data["td_C"],
                    u_ms=data["u_ms"],
                    v_ms=data["v_ms"],
                    mslp_hPa=data["mslp_hPa"],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("Skipping station %s: %s", cs, exc)

    result = ObsCollection(obs=obs_list)
    log.info(
        "CWOP fetch complete: %d stations for %s",
        len(result),
        analysis_time.strftime("%Y-%m-%d %H:%MZ"),
    )
    return result
