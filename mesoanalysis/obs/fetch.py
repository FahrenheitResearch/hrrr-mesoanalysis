"""
Fetch METAR / ASOS observations from the Iowa Environmental Mesonet (IEM)
ASOS download service and return an ObsCollection.

The IEM ASOS one-minute / routine download service is queried per-state
(network = ``{ST}_ASOS``).  We iterate over all CONUS state codes, fetch the
CSV data, and merge the results.
"""

from __future__ import annotations

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from mesoanalysis.obs.models import ObsCollection, SurfaceObs

log = logging.getLogger(__name__)

# -- CONUS state two-letter codes (50 states + DC) --------------------------
CONUS_STATES: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
]

# Default cache directory
_CACHE_DIR = Path.home() / ".cache" / "mesoanalysis" / "obs"

# IEM ASOS download base URL
_IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _build_url(state: str, start: datetime, end: datetime) -> str:
    """Build the IEM ASOS download URL for a single state network."""
    # IEM requires date components as separate parameters (year1/month1/...),
    # specific data fields, and report_type flags for routine+special METARs.
    parts = [
        f"data=tmpf",
        f"data=dwpf",
        f"data=drct",
        f"data=sknt",
        f"data=mslp",
        f"tz=Etc%2FUTC",
        f"format=onlycomma",
        f"latlon=yes",
        f"elev=yes",
        f"missing=M",
        f"trace=T",
        f"direct=no",
        f"report_type=3",
        f"report_type=4",
        f"year1={start.year}",
        f"month1={start.month}",
        f"day1={start.day}",
        f"hour1={start.hour}",
        f"minute1={start.minute}",
        f"year2={end.year}",
        f"month2={end.month}",
        f"day2={end.day}",
        f"hour2={end.hour}",
        f"minute2={end.minute}",
        f"network={state}_ASOS",
    ]
    qs = "&".join(parts)
    return f"{_IEM_BASE}?{qs}"


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _read_cache(url: str, cache_dir: Path) -> Optional[str]:
    """Return cached CSV text if it exists, else None."""
    fp = cache_dir / f"{_cache_key(url)}.csv"
    if fp.exists():
        log.debug("Cache hit: %s", fp)
        return fp.read_text(encoding="utf-8")
    return None


def _write_cache(url: str, text: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{_cache_key(url)}.csv"
    fp.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Wind conversion helpers
# ---------------------------------------------------------------------------

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


def _f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_csv(text: str) -> list[SurfaceObs]:
    """Parse IEM ASOS CSV text into a list of SurfaceObs."""
    obs_list: list[SurfaceObs] = []
    if not text or text.startswith("ERROR") or len(text) < 50:
        return obs_list

    try:
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception:
        log.warning("Failed to parse CSV chunk")
        return obs_list

    # Normalise column names (IEM uses lowercase but just in case)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"station", "lat", "lon", "valid", "tmpf", "dwpf", "drct", "sknt", "mslp"}
    if not required.issubset(set(df.columns)):
        # May have "alti" instead of "mslp"; handle below
        if "alti" in df.columns and "mslp" not in df.columns:
            # altimeter setting (inHg) -> approximate MSLP (hPa)
            df["mslp"] = pd.to_numeric(df["alti"], errors="coerce") * 33.8639
        else:
            missing_cols = required - set(df.columns)
            log.warning("CSV missing columns: %s", missing_cols)
            return obs_list

    for _, row in df.iterrows():
        try:
            # Skip rows with missing critical fields
            for field in ("tmpf", "dwpf", "drct", "sknt", "mslp"):
                val = str(row.get(field, "M")).strip()
                if val in ("M", "", "T"):
                    raise ValueError(f"Missing {field}")

            tmpf = float(row["tmpf"])
            dwpf = float(row["dwpf"])
            drct = float(row["drct"])
            sknt = float(row["sknt"])
            mslp = float(row["mslp"])
            lat = float(row["lat"])
            lon = float(row["lon"])

            t_c = _f_to_c(tmpf)
            td_c = _f_to_c(dwpf)
            u, v = _wind_components(sknt, drct)

            time_str = str(row["valid"]).strip()
            obs_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )

            obs_list.append(
                SurfaceObs(
                    station=str(row["station"]).strip(),
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
# Public API
# ---------------------------------------------------------------------------

def fetch_surface_obs(
    analysis_time: datetime,
    window_minutes: int = 20,
    states: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    timeout: int = 30,
) -> ObsCollection:
    """Fetch METAR/ASOS obs within a time window around *analysis_time*.

    Parameters
    ----------
    analysis_time : datetime
        Centre of the observation window (should be timezone-aware UTC).
    window_minutes : int
        Half-width of the time window in minutes (default +-20 min).
    states : list[str] | None
        State codes to query.  Defaults to all CONUS + DC.
    cache_dir : Path | None
        Directory for raw-CSV file cache.  ``None`` uses the default.
    timeout : int
        HTTP request timeout in seconds per state query.

    Returns
    -------
    ObsCollection
    """
    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)

    start = analysis_time - timedelta(minutes=window_minutes)
    end = analysis_time + timedelta(minutes=window_minutes)
    states = states or CONUS_STATES
    cache_dir = cache_dir or _CACHE_DIR

    all_obs: list[SurfaceObs] = []

    def _fetch_state(st: str) -> list[SurfaceObs]:
        """Fetch and parse obs for a single state (thread-safe)."""
        url = _build_url(st, start, end)
        text = _read_cache(url, cache_dir)
        if text is None:
            try:
                log.info("Fetching %s_ASOS ...", st)
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                text = resp.text
                _write_cache(url, text, cache_dir)
            except requests.RequestException as exc:
                log.warning("Failed to fetch %s_ASOS: %s", st, exc)
                return []
        return _parse_csv(text)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_state, st): st for st in states}
        for future in as_completed(futures):
            try:
                all_obs.extend(future.result())
            except Exception as exc:
                log.warning("Fetch thread error for %s: %s", futures[future], exc)

    # De-duplicate: keep the observation closest to analysis_time per station
    best: dict[str, SurfaceObs] = {}
    for o in all_obs:
        dt = abs((o.time - analysis_time).total_seconds())
        prev = best.get(o.station)
        if prev is None or dt < abs((prev.time - analysis_time).total_seconds()):
            best[o.station] = o

    result = ObsCollection(obs=list(best.values()))
    log.info(
        "Fetched %d unique stations (%d raw obs) for %s",
        len(result),
        len(all_obs),
        analysis_time.strftime("%Y-%m-%d %H:%MZ"),
    )
    return result
