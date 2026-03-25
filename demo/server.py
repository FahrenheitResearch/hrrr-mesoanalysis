"""
FastAPI demo backend for HRRR Mesoanalysis.

Serves pipeline output (tiles, manifests, point queries), real-time ASOS
observations as GeoJSON, and an optional on-demand pipeline trigger.

Run from the project root:
    cd hrrr-mesoanalysis && python -m demo.server
    cd hrrr-mesoanalysis && uvicorn demo.server:app --reload
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
STATIC_DIR = Path(__file__).resolve().parent / "static"

log = logging.getLogger("demo.server")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="HRRR Mesoanalysis Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Run discovery helpers
# ---------------------------------------------------------------------------

_RUN_PATTERN = re.compile(r"^\d{8}_\d{4}$")


def _discover_models() -> list[str]:
    """Return model subdirectories under output/ that contain run folders."""
    models = []
    if not OUTPUT_DIR.exists():
        return models
    for entry in OUTPUT_DIR.iterdir():
        if entry.is_dir() and not _RUN_PATTERN.match(entry.name):
            # Only count it as a model dir if it has at least one run subfolder
            for sub in entry.iterdir():
                if sub.is_dir() and _RUN_PATTERN.match(sub.name):
                    models.append(entry.name)
                    break
    models.sort()
    return models


def _discover_runs(model: str) -> list[str]:
    """Return run folder names for a given model, newest first."""
    model_dir = OUTPUT_DIR / model
    if not model_dir.exists():
        return []
    runs = []
    for entry in model_dir.iterdir():
        if entry.is_dir() and _RUN_PATTERN.match(entry.name):
            runs.append(entry.name)
    runs.sort(reverse=True)
    return runs


def _run_dir(model: str, run: str) -> Path:
    return OUTPUT_DIR / model / run


# ---------------------------------------------------------------------------
# Grid cache (lazy-loaded per model/run)
# ---------------------------------------------------------------------------

_grid_cache: dict[str, dict[str, np.ndarray]] = {}


def _load_grids(model: str, run: str) -> Optional[dict[str, np.ndarray]]:
    cache_key = f"{model}/{run}"
    if cache_key in _grid_cache:
        return _grid_cache[cache_key]

    npz_path = _run_dir(model, run) / "grids.npz"
    if not npz_path.exists():
        return None

    try:
        data = dict(np.load(str(npz_path), allow_pickle=False))
        _grid_cache[cache_key] = data
        log.info("Loaded grids.npz for %s (%d arrays)", cache_key, len(data))
        return data
    except Exception:
        log.exception("Failed to load grids.npz for %s", cache_key)
        return None


def _query_point(model: str, run: str, lat: float, lon: float) -> Optional[dict]:
    grids = _load_grids(model, run)
    if grids is None:
        return None

    grid_lat = grids.get("_lats", grids.get("lat"))
    grid_lon = grids.get("_lons", grids.get("lon"))
    if grid_lat is None or grid_lon is None:
        return None

    if grid_lat.ndim == 1:
        lat_idx = int(np.argmin(np.abs(grid_lat - lat)))
        lon_idx = int(np.argmin(np.abs(grid_lon - lon)))
    else:
        dist = (grid_lat - lat) ** 2 + (grid_lon - lon) ** 2
        lat_idx, lon_idx = np.unravel_index(np.argmin(dist), dist.shape)

    skip = {"lat", "lon", "_lats", "_lons"}
    result = {}
    for key, arr in grids.items():
        if key in skip or arr.ndim != 2:
            continue
        val = float(arr[lat_idx, lon_idx])
        if np.isfinite(val):
            result[key] = round(val, 2)

    return result


# ---------------------------------------------------------------------------
# Observation cache
# ---------------------------------------------------------------------------

_obs_cache: dict[str, tuple[float, dict]] = {}  # key -> (timestamp, geojson)
_OBS_CACHE_TTL = 300  # 5 minutes


def _fetch_obs_geojson(window: int) -> dict:
    cache_key = f"obs_{window}"
    cached = _obs_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _OBS_CACHE_TTL:
        return cached[1]

    from mesoanalysis.obs.fetch_multi import fetch_all_surface_obs
    from datetime import timedelta

    # Use the most recent top-of-hour for best obs coverage (ASOS reports
    # at ~:53-:56 past the hour, so querying at :15 past catches the
    # previous hour's reports within a ±30 min window)
    now = datetime.now(tz=timezone.utc)
    analysis_time = now.replace(minute=0, second=0, microsecond=0)
    if now.minute < 30:
        analysis_time -= timedelta(hours=1)
    collection = fetch_all_surface_obs(analysis_time=analysis_time, window_minutes=max(window, 30))

    features = []
    for obs in collection:
        # Convert stored C/m-s back to F/kt for display
        t_f = round(obs.t_C * 9.0 / 5.0 + 32.0, 1)
        td_f = round(obs.td_C * 9.0 / 5.0 + 32.0, 1)
        wind_speed_kt = round(obs.wind_speed_ms / 0.514444, 1)
        wind_dir = round(obs.wind_dir_deg, 0)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [obs.lon, obs.lat],
            },
            "properties": {
                "station": obs.station,
                "lat": obs.lat,
                "lon": obs.lon,
                "t_f": t_f,
                "td_f": td_f,
                "wind_speed_kt": wind_speed_kt,
                "wind_dir": wind_dir,
                "mslp_hPa": round(obs.mslp_hPa, 1) if np.isfinite(obs.mslp_hPa) else None,
                "time": obs.time.isoformat(),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    _obs_cache[cache_key] = (time.time(), geojson)
    return geojson


# ---------------------------------------------------------------------------
# Pipeline runner state
# ---------------------------------------------------------------------------

_pipeline_lock = threading.Lock()
_pipeline_running = False
_pipeline_job_id: Optional[str] = None
_pipeline_status: str = "idle"


def _run_pipeline_thread(model: str):
    global _pipeline_running, _pipeline_status
    try:
        _pipeline_status = "running"
        from mesoanalysis.pipeline import run as run_pipeline

        now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        run_pipeline(analysis_time=now, model=model)
        _pipeline_status = "completed"
    except Exception as exc:
        log.exception("Pipeline run failed")
        _pipeline_status = f"failed: {exc}"
    finally:
        with _pipeline_lock:
            _pipeline_running = False


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def list_runs():
    """List available models and their runs."""
    from fastapi.responses import JSONResponse

    models = _discover_models()
    result = {}
    for m in models:
        result[m] = _discover_runs(m)

    # Also include legacy top-level run dirs (output/YYYYMMDD_HHMM)
    if OUTPUT_DIR.exists():
        legacy = []
        for entry in OUTPUT_DIR.iterdir():
            if entry.is_dir() and _RUN_PATTERN.match(entry.name):
                legacy.append(entry.name)
        if legacy:
            legacy.sort(reverse=True)
            result["_legacy"] = legacy

    # Add a "latest" field for the frontend
    latest = None
    for m in ["hrrr", "rap", "nam"]:
        runs = result.get(m, [])
        if runs:
            latest = {"model": m, "run_id": runs[0]}
            break
    result["latest"] = latest

    return JSONResponse(content=result)


@app.get("/api/manifest/{model}/{run}")
def get_manifest(model: str, run: str):
    """Return manifest.json for a specific model run."""
    manifest_path = _run_dir(model, run) / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    return JSONResponse(content=json.loads(manifest_path.read_text(encoding="utf-8")))


@app.get("/api/tiles/{model}/{run}/{param}.png")
def get_tile(model: str, run: str, param: str):
    """Serve a transparent overlay PNG tile for a parameter."""
    tile_path = _run_dir(model, run) / "tiles" / f"{param}.png"
    if not tile_path.exists():
        raise HTTPException(status_code=404, detail=f"Tile not found: {param}")
    return FileResponse(
        tile_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/api/point/{model}/{run}")
def point_query(model: str, run: str, lat: float = Query(...), lon: float = Query(...)):
    """Query all parameter values at a given lat/lon from grids.npz."""
    runs = _discover_runs(model)
    if run not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    values = _query_point(model, run, lat, lon)
    if values is None:
        return {"lat": lat, "lon": lon, "model": model, "run": run,
                "error": "Grid data not available", "values": None}

    return {"lat": lat, "lon": lon, "model": model, "run": run, "values": values}


@app.get("/api/obs")
def get_observations(window: int = Query(default=20, ge=1, le=120)):
    """Fetch current ASOS observations as GeoJSON FeatureCollection."""
    try:
        geojson = _fetch_obs_geojson(window)
        return geojson
    except Exception as exc:
        log.exception("Failed to fetch observations")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/run")
def trigger_run(model: str = Query(default="hrrr")):
    """Trigger a pipeline run for the most recent hour. Returns immediately."""
    global _pipeline_running, _pipeline_job_id

    with _pipeline_lock:
        if _pipeline_running:
            return {"status": "already_running", "job_id": _pipeline_job_id}
        _pipeline_running = True
        _pipeline_job_id = str(uuid.uuid4())[:8]

    thread = threading.Thread(target=_run_pipeline_thread, args=(model,), daemon=True)
    thread.start()

    return {"status": "started", "job_id": _pipeline_job_id, "model": model}


@app.get("/api/run/status")
def run_status():
    """Check if a pipeline run is in progress."""
    return {
        "running": _pipeline_running,
        "job_id": _pipeline_job_id,
        "status": _pipeline_status,
    }


# ---------------------------------------------------------------------------
# Static files (must be last — catches all remaining paths)
# ---------------------------------------------------------------------------

STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Create a minimal index.html if static dir is empty
_index = STATIC_DIR / "index.html"
if not _index.exists():
    _index.write_text(
        '<!DOCTYPE html><html><head><title>HRRR Mesoanalysis</title></head>'
        '<body><h1>HRRR Mesoanalysis Demo</h1>'
        '<p>API endpoints:</p><ul>'
        '<li><a href="/api/runs">/api/runs</a> &mdash; list runs</li>'
        '<li>/api/manifest/{model}/{run} &mdash; run manifest</li>'
        '<li>/api/tiles/{model}/{run}/{param}.png &mdash; overlay tiles</li>'
        '<li>/api/point/{model}/{run}?lat=X&amp;lon=Y &mdash; point query</li>'
        '<li><a href="/api/obs">/api/obs</a> &mdash; live observations (GeoJSON)</li>'
        '<li>/api/run/status &mdash; pipeline status</li>'
        '</ul></body></html>',
        encoding="utf-8",
    )

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ---------------------------------------------------------------------------
# CLI entry point: python -m demo.server
# ---------------------------------------------------------------------------

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="HRRR Mesoanalysis demo server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    models = _discover_models()
    print(f"HRRR Mesoanalysis Demo Server")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    if models:
        for m in models:
            runs = _discover_runs(m)
            print(f"  {m:12s} : {len(runs)} run(s)" + (f" (latest: {runs[0]})" if runs else ""))
    else:
        print("  No model runs found in output/")
    print(f"\n  http://localhost:{args.port}")
    print(f"  http://localhost:{args.port}/docs  (Swagger UI)\n")

    uvicorn.run(
        "demo.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
