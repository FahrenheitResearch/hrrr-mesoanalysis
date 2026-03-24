"""
Unified multi-source surface observation fetcher.

Aggregates observations from all available networks:
  1. IEM ASOS (existing — fetch.py)
  2. IEM Mesonets / RWIS / AWOS / COOP (fetch_mesonets.py)
  3. NWS API (fetch_nws.py)
  4. NDBC buoys (fetch_ndbc.py)
  5. CWOP / APRS citizen stations (fetch_cwop.py)

All sources are fetched concurrently using threads.  Results are merged
and deduplicated by station ID, with priority given to higher-quality
networks when duplicate IDs exist.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, Optional

from mesoanalysis.obs.models import ObsCollection, SurfaceObs

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

# Priority: lower number = higher trust.  When two sources report the same
# station ID, the one with lower priority number wins.
_SOURCE_PRIORITY = {
    "asos": 1,
    "nws": 2,
    "mesonet": 3,
    "ndbc": 4,
    "cwop": 5,
}


def _fetch_asos(analysis_time: datetime, window_minutes: int) -> tuple[str, ObsCollection]:
    from mesoanalysis.obs.fetch import fetch_surface_obs
    return "asos", fetch_surface_obs(analysis_time, window_minutes=window_minutes)


def _fetch_mesonets(analysis_time: datetime, window_minutes: int) -> tuple[str, ObsCollection]:
    from mesoanalysis.obs.fetch_mesonets import fetch_mesonet_obs
    # Use the same tight window as ASOS — DCP/RWIS stations that haven't
    # reported within ±20 min are stale and should not be used.
    return "mesonet", fetch_mesonet_obs(analysis_time, window_minutes=window_minutes)


def _fetch_nws(analysis_time: datetime, window_minutes: int) -> tuple[str, ObsCollection]:
    from mesoanalysis.obs.fetch_nws import fetch_nws_obs
    return "nws", fetch_nws_obs(analysis_time, window_minutes=window_minutes)


def _fetch_ndbc(analysis_time: datetime, window_minutes: int) -> tuple[str, ObsCollection]:
    from mesoanalysis.obs.fetch_ndbc import fetch_ndbc_obs
    # Buoys report hourly — use ±60 min window.  They mainly contribute
    # pressure and coastal T/Td.  Pressure evolves slowly so this is safe.
    return "ndbc", fetch_ndbc_obs(analysis_time, window_minutes=60)


def _fetch_cwop(analysis_time: datetime, window_minutes: int) -> tuple[str, ObsCollection]:
    from mesoanalysis.obs.fetch_cwop import fetch_cwop_obs
    return "cwop", fetch_cwop_obs(analysis_time, window_minutes=max(window_minutes, 30))


# Default sources for real-time mesoanalysis (±20 min window).
#
# Only ASOS and NDBC are included — these are the networks that reliably
# report within a tight time window.  DCP/RWIS/COOP stations report on
# 1-2 hour cycles and produce ~0 obs within ±20 min.  NWS API is
# redundant with IEM ASOS and frequently times out.  CWOP is citizen-grade.
# All can be enabled via sources=["all"].
_SOURCES: list[Callable] = [
    _fetch_asos,
    _fetch_ndbc,
]

_ALL_SOURCES: list[Callable] = [
    _fetch_asos,
    _fetch_mesonets,
    _fetch_nws,
    _fetch_ndbc,
    _fetch_cwop,
]


# ---------------------------------------------------------------------------
# Merging / deduplication
# ---------------------------------------------------------------------------

def _merge_obs(
    source_collections: list[tuple[str, ObsCollection]],
    analysis_time: datetime,
) -> ObsCollection:
    """Merge observations from multiple sources, deduplicating by station ID.

    When the same station ID appears from multiple sources, the observation
    from the highest-priority source is kept.  When the same station ID
    appears within a single source, the observation closest to analysis_time
    is kept.
    """
    # station_id -> (priority, SurfaceObs)
    best: dict[str, tuple[int, SurfaceObs]] = {}

    for source_name, collection in source_collections:
        priority = _SOURCE_PRIORITY.get(source_name, 99)
        for obs in collection:
            key = obs.station
            existing = best.get(key)
            if existing is None:
                best[key] = (priority, obs)
            else:
                ex_prio, ex_obs = existing
                if priority < ex_prio:
                    # Higher-priority source wins
                    best[key] = (priority, obs)
                elif priority == ex_prio:
                    # Same source — keep the one closest to analysis_time
                    dt_new = abs((obs.time - analysis_time).total_seconds())
                    dt_old = abs((ex_obs.time - analysis_time).total_seconds())
                    if dt_new < dt_old:
                        best[key] = (priority, obs)

    merged = [obs for _, obs in best.values()]
    return ObsCollection(obs=merged)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_surface_obs(
    analysis_time: datetime,
    window_minutes: int = 20,
    sources: Optional[list[str]] = None,
    skip_slow: bool = False,
) -> ObsCollection:
    """Fetch surface observations from all available networks.

    Parameters
    ----------
    analysis_time : datetime
        Centre of the observation window (should be timezone-aware UTC).
    window_minutes : int
        Half-width of the time window in minutes (default +-20 min).
    sources : list[str] | None
        Restrict to a subset of sources by name.  Valid names:
        ``"asos"``, ``"mesonet"``, ``"nws"``, ``"ndbc"``, ``"cwop"``.
        ``None`` means all sources.
    skip_slow : bool
        If True, skip the NWS API and CWOP sources which are significantly
        slower than the others due to rate limiting / many HTTP requests.

    Returns
    -------
    ObsCollection
        Merged, deduplicated observations from all requested sources.
    """
    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)

    # Decide which sources to run
    active_sources = list(_SOURCES)
    if sources is not None and "all" in sources:
        active_sources = list(_ALL_SOURCES)
    elif skip_slow:
        active_sources = [s for s in active_sources
                          if s not in (_fetch_nws, _fetch_cwop)]
    if sources is not None and "all" not in sources:
        name_set = set(sources)
        name_map = {
            "asos": _fetch_asos,
            "mesonet": _fetch_mesonets,
            "nws": _fetch_nws,
            "ndbc": _fetch_ndbc,
            "cwop": _fetch_cwop,
        }
        active_sources = [name_map[n] for n in name_set if n in name_map]

    log.info(
        "Multi-source fetch: %d sources for %s",
        len(active_sources),
        analysis_time.strftime("%Y-%m-%d %H:%MZ"),
    )

    # Fetch all sources concurrently
    results: list[tuple[str, ObsCollection]] = []

    with ThreadPoolExecutor(max_workers=len(active_sources)) as pool:
        futures = {
            pool.submit(fn, analysis_time, window_minutes): fn.__name__
            for fn in active_sources
        }
        for future in as_completed(futures):
            fn_name = futures[future]
            try:
                source_name, collection = future.result()
                log.info("  %s: %d obs", source_name, len(collection))
                results.append((source_name, collection))
            except Exception as exc:
                log.error("Source %s failed: %s", fn_name, exc)

    # Merge and deduplicate
    merged = _merge_obs(results, analysis_time)

    # Summary
    total_raw = sum(len(c) for _, c in results)
    log.info(
        "Multi-source total: %d raw -> %d merged unique stations",
        total_raw, len(merged),
    )

    return merged
