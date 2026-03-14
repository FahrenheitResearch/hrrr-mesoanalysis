"""SRH benchmark: per-column loop vs grid-native Rust on full HRRR."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from datetime import datetime
from metrust._metrust import calc as _calc
from mesoanalysis.hrrr.ingest import load_hrrr

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading HRRR data...")
data = load_hrrr(datetime(2026, 3, 10, 18, 0))
ny, nx = data.ny, data.nx
nz = len(data.levels_mb)
print(f"Grid: {ny}x{nx}, {nz} levels, {ny*nx:,} columns")

u_3d = data.u.ravel().astype(np.float64)
v_3d = data.v.ravel().astype(np.float64)
h_3d = data.h_agl_m.ravel().astype(np.float64)

# Warm up
_calc.compute_srh(u_3d, v_3d, h_3d, nx, ny, nz, 1000.0)

# --- Grid-native SRH ---
times = []
for i in range(5):
    t0 = time.perf_counter()
    srh = np.asarray(_calc.compute_srh(u_3d, v_3d, h_3d, nx, ny, nz, 1000.0))
    times.append(time.perf_counter() - t0)
dt_native = np.mean(times)
print(f"\nGrid-native 0-1km SRH: {dt_native*1000:.1f} ms ({ny*nx:,} columns)")
print(f"  Rate: {ny*nx/dt_native:,.0f} columns/sec")

# --- Per-column loop (16k sample) ---
n_test = 16_000
t1 = time.perf_counter()
for idx in range(n_test):
    j = idx // nx
    i = idx % nx
    u_col = np.array([data.u[k, j, i] for k in range(nz)], dtype=np.float64)
    v_col = np.array([data.v[k, j, i] for k in range(nz)], dtype=np.float64)
    h_col = np.array([data.h_agl_m[k, j, i] for k in range(nz)], dtype=np.float64)
    _calc.storm_relative_helicity(u_col, v_col, h_col, 1000.0, 0.0, 0.0)
dt_loop = (time.perf_counter() - t1)
per_col_us = dt_loop / n_test * 1e6
extrapolated = dt_loop / n_test * ny * nx
print(f"\nPer-column loop: {n_test:,} in {dt_loop*1000:.0f} ms ({per_col_us:.1f} us/col)")
print(f"  Extrapolated full grid: {extrapolated:.1f}s")

srh_2d = srh.reshape(ny, nx)
speedup_grid_vs_loop = extrapolated / dt_native

results = f"""SRH Benchmark
{'='*45}
Grid: {ny}x{nx} = {ny*nx:,} columns, {nz} levels

Grid-native (Rust):   {dt_native*1000:.1f} ms for {ny*nx:,} columns
Per-column loop:      {dt_loop*1000:.0f} ms for {n_test:,} columns -> {extrapolated:.1f}s full grid
Speedup (grid-native vs per-column): {speedup_grid_vs_loop:.0f}x
Max SRH: {np.nanmax(srh_2d):.0f} m^2/s^2
"""
print(results)

with open(os.path.join(OUTPUT_DIR, "bench_srh.txt"), "w") as f:
    f.write(results)
print(f"Saved to {OUTPUT_DIR}/bench_srh.txt")
