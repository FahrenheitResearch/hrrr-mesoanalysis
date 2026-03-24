"""
Minimal development server for the HRRR Mesoanalysis web viewer.

Serves:
  /              -> web/ directory (HTML/CSS/JS)
  /output/...    -> output/ directory (parameter PNGs)
  /api/latest    -> JSON with available runs and latest run name
  /api/point     -> JSON with all parameter values at a lat/lon point

Usage:
  python web/serve.py
  python -m web.serve
  python web/serve.py --port 8080

Runs on http://localhost:8080 by default.
"""

import argparse
import json
import os
import sys
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve project root (parent of web/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
WEB_DIR = SCRIPT_DIR


# ---------------------------------------------------------------------------
# Grid data cache (lazy-loaded)
# ---------------------------------------------------------------------------

_grid_cache = {}  # run_name -> {param: ndarray, "lat": ndarray, "lon": ndarray}


def _load_grids(run_name):
    """
    Attempt to load grid data for point queries.

    Tries (in order):
      1. output/{run}/grids.npz  — full grid archive
      2. Re-run the pipeline params in-memory (not implemented here)

    Returns dict of {param_name: 2d-array, "lat": 2d, "lon": 2d} or None.
    """
    if run_name in _grid_cache:
        return _grid_cache[run_name]

    npz_path = OUTPUT_DIR / run_name / "grids.npz"
    print(f"[serve] Looking for grids at: {npz_path} (exists={npz_path.exists()})")
    if npz_path.exists():
        try:
            import numpy as np
            data = dict(np.load(str(npz_path)))
            _grid_cache[run_name] = data
            print(f"[serve] Loaded grids.npz for {run_name} ({len(data)} arrays)")
            return data
        except Exception as e:
            print(f"[serve] Failed to load grids.npz: {e}", flush=True)
            import traceback
            traceback.print_exc()

    return None


def _query_point(run_name, lat, lon):
    """
    Query all parameter values at a given lat/lon.
    Returns dict of {param: float} or None.
    """
    grids = _load_grids(run_name)
    if grids is None:
        return None

    try:
        import numpy as np

        # Support both key conventions: lat/lon and _lats/_lons
        grid_lat = grids.get("lat", grids.get("_lats"))
        grid_lon = grids.get("lon", grids.get("_lons"))
        if grid_lat is None or grid_lon is None:
            return None

        # Find nearest grid point
        if grid_lat.ndim == 2:
            # Curvilinear grid (2D lat/lon)
            dist = (grid_lat - lat) ** 2 + (grid_lon - lon) ** 2
            idx = np.unravel_index(np.argmin(dist), dist.shape)
        else:
            # Regular 1D grid (from web export reprojection)
            lat_idx = int(np.argmin(np.abs(grid_lat - lat)))
            lon_idx = int(np.argmin(np.abs(grid_lon - lon)))
            idx = (lat_idx, lon_idx)

        # Extract values for all parameters
        skip_keys = {"lat", "lon", "_lats", "_lons"}
        result = {}
        for key, arr in grids.items():
            if key in skip_keys:
                continue
            if arr.ndim == 2:
                val = float(arr[idx[0], idx[1]])
                if not (np.isnan(val) or np.isinf(val)):
                    result[key] = round(val, 2)

        return result

    except Exception as e:
        print(f"[serve] Point query error: {e}")
        return None


# ---------------------------------------------------------------------------
# Available runs discovery
# ---------------------------------------------------------------------------

_RUN_PATTERN = re.compile(r"^\d{8}_\d{4}$")


def _discover_runs():
    """Scan output/ for valid run directories, sorted newest first."""
    if not OUTPUT_DIR.exists():
        return []
    runs = []
    for entry in OUTPUT_DIR.iterdir():
        if entry.is_dir() and _RUN_PATTERN.match(entry.name):
            # Check that it has at least one PNG
            pngs = list(entry.glob("*.png"))
            if pngs:
                runs.append(entry.name)
    runs.sort(reverse=True)
    return runs


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class MesoHandler(SimpleHTTPRequestHandler):
    """Custom handler that routes API requests and serves static files."""

    def __init__(self, *args, **kwargs):
        # Set web/ as the serving directory for static files
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ---- API endpoints ----
        if path == "/api/latest":
            self._handle_latest()
            return

        if path == "/api/point":
            try:
                self._handle_point(parsed.query)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json(json.dumps({"error": str(e)}), status=500)
            return

        # ---- Serve output/ files ----
        if path.startswith("/output/"):
            rel_path = path[len("/output/"):]
            # Security: prevent path traversal
            rel_path = rel_path.replace("\\", "/")
            if ".." in rel_path:
                self.send_error(403, "Forbidden")
                return
            file_path = OUTPUT_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                self._serve_file(file_path)
                return
            else:
                self.send_error(404, "Not Found")
                return

        # ---- Serve web/ static files (default) ----
        super().do_GET()

    def _handle_latest(self):
        runs = _discover_runs()
        latest = runs[0] if runs else None
        body = json.dumps({"latest": latest, "runs": runs})
        self._send_json(body)

    def _handle_point(self, query_string):
        params = parse_qs(query_string)
        try:
            lat = float(params["lat"][0])
            lon = float(params["lon"][0])
            run = params.get("run", [None])[0]
        except (KeyError, ValueError, IndexError):
            self.send_error(400, "Bad request: need lat, lon, run parameters")
            return

        if run is None:
            runs = _discover_runs()
            run = runs[0] if runs else None

        if run is None:
            self._send_json(json.dumps({"error": "No runs available"}), status=404)
            return

        values = None
        try:
            import numpy as np
            npz = OUTPUT_DIR / run / "grids.npz"
            if npz.exists():
                data = dict(np.load(str(npz), allow_pickle=False))
                grid_lat = data.get("_lats", data.get("lat"))
                grid_lon = data.get("_lons", data.get("lon"))
                if grid_lat is not None and grid_lon is not None:
                    if grid_lat.ndim == 1:
                        lat_idx = int(np.argmin(np.abs(grid_lat - lat)))
                        lon_idx = int(np.argmin(np.abs(grid_lon - lon)))
                    else:
                        dist = (grid_lat - lat)**2 + (grid_lon - lon)**2
                        lat_idx, lon_idx = np.unravel_index(np.argmin(dist), dist.shape)
                    skip = {"lat", "lon", "_lats", "_lons"}
                    values = {}
                    for k, arr in data.items():
                        if k in skip or arr.ndim != 2:
                            continue
                        v = float(arr[lat_idx, lon_idx])
                        if np.isfinite(v):
                            values[k] = round(v, 2)
        except Exception as exc:
            self.log_message("Point query error: %s", exc)
        if not values:
            self._send_json(json.dumps({
                "lat": lat, "lon": lon, "run": run,
                "error": "Grid data not available.",
                "values": None
            }))
        else:
            self._send_json(json.dumps({
                "lat": lat, "lon": lon, "run": run,
                "values": values
            }))

    def _send_json(self, body, status=200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._add_cors()
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, file_path):
        """Serve a file from the output directory."""
        ext = file_path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".json": "application/json",
            ".npz": "application/octet-stream",
            ".csv": "text/csv",
        }
        ct = content_types.get(ext, "application/octet-stream")

        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self._add_cors()
            # Cache images for 5 minutes
            if ext in (".png", ".jpg", ".jpeg"):
                self.send_header("Cache-Control", "public, max-age=300")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def _add_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self._add_cors()
        self.end_headers()

    def log_message(self, format, *args):
        """Quieter logging — only log non-200 or API requests."""
        status = args[1] if len(args) > 1 else ""
        path = args[0] if args else ""
        if "/api/" in str(path) or str(status) != "200":
            super().log_message(format, *args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HRRR Mesoanalysis development server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--bind", default="0.0.0.0", help="Address to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"HRRR Mesoanalysis Server")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Web dir      : {WEB_DIR}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print()

    runs = _discover_runs()
    if runs:
        print(f"  Available runs: {', '.join(runs)}")
        print(f"  Latest run   : {runs[0]}")
    else:
        print("  No runs found in output/")

    print()
    print(f"  Serving on http://{args.bind}:{args.port}")
    print(f"  Open http://localhost:{args.port} in your browser")
    print()

    server = HTTPServer((args.bind, args.port), MesoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
