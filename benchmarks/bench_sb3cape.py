"""SB3CAPE benchmark on full HRRR grid — the thing nobody attempts with MetPy."""

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
print(f"Grid: {ny}x{nx}, {nz} levels, {ny*nx:,} columns\n")

p_3d = data.p_Pa.ravel().astype(np.float64)
t_3d = data.t_C.ravel().astype(np.float64)
q_3d = data.q_kgkg.ravel().astype(np.float64)
h_3d = data.h_agl_m.ravel().astype(np.float64)
psfc = data.psfc_Pa.ravel().astype(np.float64)
t2 = data.t2m_K.ravel().astype(np.float64)

td2_C = data.td2m_K - 273.15
e_sat = 6.112 * np.exp(17.67 * td2_C / (td2_C + 243.5))
psfc_hPa = data.psfc_Pa / 100.0
q2 = (0.622 * e_sat / (psfc_hPa - e_sat)).ravel().astype(np.float64)

# Warm up
_calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2, nx, ny, nz, "surface", top_m=3000.0)

# --- SBCAPE (full column) ---
times_full = []
for i in range(3):
    t0 = time.perf_counter()
    sb = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2, nx, ny, nz, "surface")
    elapsed = time.perf_counter() - t0
    times_full.append(elapsed)
    cape_full = np.array(sb[0]).reshape(ny, nx)
    print(f"  SBCAPE run {i+1}: {elapsed*1000:.0f}ms  max={np.nanmax(cape_full):.0f} J/kg")
dt_full = np.mean(times_full)

# --- SB3CAPE (0-3km cap) ---
times_3km = []
for i in range(3):
    t0 = time.perf_counter()
    sb3 = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2, nx, ny, nz, "surface", top_m=3000.0)
    elapsed = time.perf_counter() - t0
    times_3km.append(elapsed)
    cape_3km = np.array(sb3[0]).reshape(ny, nx)
    print(f"  SB3CAPE run {i+1}: {elapsed*1000:.0f}ms  max={np.nanmax(cape_3km):.0f} J/kg")
dt_3km = np.mean(times_3km)

# --- MLCAPE ---
times_ml = []
for i in range(3):
    t0 = time.perf_counter()
    ml = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2, nx, ny, nz, "mixed_layer")
    elapsed = time.perf_counter() - t0
    times_ml.append(elapsed)
dt_ml = np.mean(times_ml)
print(f"  MLCAPE avg: {dt_ml*1000:.0f}ms")

# --- MUCAPE ---
times_mu = []
for i in range(3):
    t0 = time.perf_counter()
    mu = _calc.compute_cape_cin(p_3d, t_3d, q_3d, h_3d, psfc, t2, q2, nx, ny, nz, "most_unstable")
    elapsed = time.perf_counter() - t0
    times_mu.append(elapsed)
dt_mu = np.mean(times_mu)
print(f"  MUCAPE avg: {dt_mu*1000:.0f}ms")

# MetPy extrapolation: ~3.7ms/column from bench_cape
metpy_per_col_ms = 3.7
metpy_full_s = metpy_per_col_ms * ny * nx / 1000
metpy_full_min = metpy_full_s / 60

results = f"""SB3CAPE Grid Benchmark
{'='*45}
Grid: {ny}x{nx} = {ny*nx:,} columns, {nz} levels

metrust grid-native timings (avg of 3 runs):
  SBCAPE:  {dt_full*1000:.0f}ms   max={np.nanmax(cape_full):.0f} J/kg
  SB3CAPE: {dt_3km*1000:.0f}ms   max={np.nanmax(cape_3km):.0f} J/kg
  MLCAPE:  {dt_ml*1000:.0f}ms
  MUCAPE:  {dt_mu*1000:.0f}ms

MetPy extrapolated (from per-column benchmark):
  ~{metpy_full_min:.0f} min for {ny*nx:,} columns

Speedup (SB3CAPE): ~{metpy_full_s/dt_3km:,.0f}x
Points with SB3CAPE > 50 J/kg: {np.sum(cape_3km > 50):,}
"""
print(results)

with open(os.path.join(OUTPUT_DIR, "bench_sb3cape.txt"), "w") as f:
    f.write(results)
print(f"Saved to {OUTPUT_DIR}/bench_sb3cape.txt")
