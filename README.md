# HRRR Mesoanalysis

High-resolution mesoscale analysis system that blends real surface observations into HRRR model data using Barnes objective analysis, then computes a full suite of severe weather parameters via [metrust](https://github.com/FahrenheitResearch/metrust-py).

This is similar to what the [SPC Mesoanalysis](https://www.spc.noaa.gov/exper/mesoanalysis/) does, but at **3km HRRR resolution** instead of 13km RAP — roughly 19x more grid columns.

## What it does

1. **Loads HRRR** surface + 22 pressure levels via [Herbie](https://github.com/blaylockbk/Herbie)
2. **Fetches ~1,700 ASOS observations** from the Iowa Environmental Mesonet
3. **Quality controls** observations against the HRRR first guess (range + gross error checks)
4. **Barnes objective analysis** blends obs into the model grid (multi-pass, Gaussian-weighted)
5. **Computes parameters** using metrust's Rust engine at full HRRR resolution (~1.9M grid points)
6. **Renders CONUS maps** for all parameters

## Output fields

| Category | Parameters |
|---|---|
| Thermodynamic | SBCAPE, SBCIN, MLCAPE, MUCAPE, SB3CAPE, LCL, Theta-e (sfc/850), Wet Bulb, Lapse Rates, Mixing Ratio, PW |
| Kinematic | 0-1/3/6km Bulk Shear, 0-1/3km SRH |
| Composite | Significant Tornado Parameter (STP), Supercell Composite (SCP) |
| Surface | 2m Temperature, 2m Dewpoint |

## Quick start

```bash
pip install metrust herbie-data cartopy scipy matplotlib requests
git clone https://github.com/FahrenheitResearch/hrrr-mesoanalysis.git
cd hrrr-mesoanalysis

# Run for any HRRR analysis time
python -m mesoanalysis 2025-05-10T18:00
```

Output PNGs land in `output/YYYYMMDD_HHMM/`.

## Why metrust

Computing SBCAPE across 1.9M HRRR columns with 22 pressure levels takes **~400ms** with metrust vs an estimated **~70 minutes** with MetPy. SRH is even more dramatic: 19ms vs 30+ minutes. This makes full-resolution mesoanalysis practical as a near-real-time product.

## Configuration

Edit `mesoanalysis/config.py` to tune:

- **Domain**: CONUS extent, map projection
- **Barnes**: `kappa` (smoothing scale), `gamma` (convergence), `passes`
- **Obs**: time window for observation matching
- **Output**: figure size, DPI

## Architecture

```
mesoanalysis/
  config.py          # Central configuration
  pipeline.py        # Orchestrator (run all 8 steps)
  __main__.py        # CLI entry point
  hrrr/
    ingest.py        # HRRR data loading via Herbie
  obs/
    fetch.py         # IEM ASOS observation fetching
    models.py        # SurfaceObs / ObsCollection dataclasses
    qc.py            # Range + gross error quality control
  analysis/
    barnes.py        # Multi-pass Barnes objective analysis
    fields.py        # Merge analyzed fields back into HRRR
  params/
    thermodynamic.py # CAPE, theta-e, wet bulb, lapse rates, PW
    kinematic.py     # Bulk shear, SRH
    composite.py     # STP, SCP
  plotting/
    maps.py          # Map factory (Lambert Conformal)
    styles.py        # Color scales and contour levels
    render.py        # Rendering engine
```

## License

MIT
