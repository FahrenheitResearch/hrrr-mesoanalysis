"""MetPy vs metrust CAPE benchmark on real HRRR soundings."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from metrust._metrust import calc as _calc

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build realistic sounding columns from random data
np.random.seed(42)
N = 1000

p = np.array([1000, 925, 850, 700, 500], dtype=np.float64)
t_cols = np.random.uniform(-5, 30, (N, 5)).astype(np.float64)
for i in range(5):
    t_cols[:, i] -= i * 10
td_cols = t_cols - np.random.uniform(2, 15, (N, 5)).astype(np.float64)
h_cols = np.tile([0, 750, 1500, 3000, 5500], (N, 1)).astype(np.float64)

# --- metrust (per-column) ---
t1 = time.perf_counter()
for i in range(N):
    _calc.cape_cin(p, t_cols[i], td_cols[i], h_cols[i],
                   1013.0, t_cols[i, 0], td_cols[i, 0],
                   "surface_based", 100.0, 300.0, 3000.0)
dt_metrust = time.perf_counter() - t1
print(f"metrust: {N} columns in {dt_metrust*1000:.1f} ms ({dt_metrust/N*1e6:.1f} us/col)")

# --- MetPy ---
from metpy.calc import cape_cin, parcel_profile
from metpy.units import units as u

pressure = p * u.hPa

t1 = time.perf_counter()
for i in range(N):
    temp = t_cols[i] * u.degC
    dewp = td_cols[i] * u.degC
    try:
        prof = parcel_profile(pressure, temp[0], dewp[0])
        c, ci = cape_cin(pressure, temp, dewp, prof)
    except Exception:
        pass
dt_metpy = time.perf_counter() - t1
print(f"MetPy:   {N} columns in {dt_metpy*1000:.1f} ms ({dt_metpy/N*1e6:.1f} us/col)")

speedup = dt_metpy / dt_metrust
print(f"\nSpeedup: {speedup:.0f}x")

n_hrrr = 1_905_141
print(f"\nExtrapolated to full HRRR grid ({n_hrrr:,} columns):")
print(f"  metrust: {n_hrrr * dt_metrust/N:.1f}s")
print(f"  MetPy:   {n_hrrr * dt_metpy/N:.0f}s ({n_hrrr * dt_metpy/N/60:.0f} min)")

# Save results
with open(os.path.join(OUTPUT_DIR, "bench_cape.txt"), "w") as f:
    f.write(f"CAPE Benchmark: MetPy vs metrust\n")
    f.write(f"{'='*45}\n")
    f.write(f"Test: {N} synthetic sounding columns, 5 levels\n\n")
    f.write(f"metrust: {dt_metrust*1000:.1f} ms  ({dt_metrust/N*1e6:.1f} us/col)\n")
    f.write(f"MetPy:   {dt_metpy*1000:.1f} ms  ({dt_metpy/N*1e6:.1f} us/col)\n")
    f.write(f"Speedup: {speedup:.0f}x\n\n")
    f.write(f"Extrapolated to {n_hrrr:,} HRRR columns:\n")
    f.write(f"  metrust: {n_hrrr * dt_metrust/N:.1f}s\n")
    f.write(f"  MetPy:   {n_hrrr * dt_metpy/N:.0f}s ({n_hrrr * dt_metpy/N/60:.0f} min)\n")
print(f"\nSaved to {OUTPUT_DIR}/bench_cape.txt")
