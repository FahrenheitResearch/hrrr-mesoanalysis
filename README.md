# mesoanalysis

Multi-model mesoscale analysis engine. Corrects NWP model surface fields with real-time observations using Barnes objective analysis, then computes 46 severe weather, fire weather, and heat stress parameters across the full grid using a Rust computation backend.

Supports HRRR (3km), RAP (13km), and NAM (3km CONUS nest). Outputs web-ready transparent PNG overlays with JSON manifests for direct integration into any map-based frontend.

## Quick start

```python
from datetime import datetime
from mesoanalysis.pipeline import run

# HRRR mesoanalysis (default)
run(datetime(2026, 3, 24, 18, 0))

# RAP mesoanalysis
run(datetime(2026, 3, 24, 18, 0), model="rap")
```

CLI:

```bash
python -m mesoanalysis 2026-03-24T18:00
python -m mesoanalysis 2026-03-24T18:00 --model rap
```

## Supported models

| Model | Resolution | Grid | Update cycle | Notes |
|-------|-----------|------|-------------|-------|
| HRRR  | 3 km      | 1799 x 1059 | Hourly | Default. Best resolution for mesoscale features |
| RAP   | 13 km     | 451 x 337   | Hourly | Same obs correction, coarser background |
| NAM   | 3-12 km   | CONUS nest   | 6-hourly | Uses RH instead of dewpoint at pressure levels |

The system uses a generic `ModelData` container. Any model that can populate the required surface and pressure-level fields will work.

## How it works

1. **Model ingest** -- Herbie pulls GRIB2 from NOMADS/AWS. Surface fields (T, Td, wind, pressure, DSWRF, cloud cover) plus 21-22 pressure levels of T, moisture, height, and wind.

2. **Multi-source obs** -- 1,600+ real-time ASOS stations within +/-20 minutes via Iowa Environmental Mesonet, plus NDBC buoys for coastal coverage.

3. **3-pass quality control** -- range checks, gross-error checks against the model first guess (KDTree interpolation on the native curvilinear grid), and spatial buddy checks. SPC-grade thresholds tuned per model resolution.

4. **Barnes objective analysis** -- two-pass successive correction. Kappa tuned to station density per model (1.8e9 for HRRR, 5.0e9 for RAP). Computed via metrust Rust backend.

5. **Parameter computation** -- corrected surface fields and 3-D profiles feed into metrust for vectorized Rust-native computation of 46 parameters across the full grid.

6. **Web export** -- each parameter rendered as a transparent RGBA PNG on EPSG:4326, with `manifest.json` (bounds + metadata) and `grids.npz` (raw arrays for point queries).

## Output

```
output/hrrr/20260324_1800/
    manifest.json       # bounds, parameter metadata, file paths
    grids.npz           # compressed float32 arrays for point queries
    tiles/
        sbcape.png      # transparent RGBA overlays (37 parameters)
        ship.png
        wbgt.png
        ...
    sbcape.png          # static matplotlib maps with basemap (37 maps)
    ...
```

The `tiles/` PNGs are projection-free overlays for web maps. The root PNGs are standalone static maps with borders and colorbars.

**manifest.json:**

```json
{
  "analysis_time": "2026-03-24T18:00:00",
  "bounds": [[24.0, -125.0], [50.0, -66.0]],
  "grid_shape": [867, 1967],
  "parameters": {
    "sbcape": {
      "display_name": "SBCAPE",
      "units": "J/kg",
      "min": 0.0,
      "max": 5842.31,
      "png_path": "tiles/sbcape.png"
    }
  }
}
```

**Web map integration:**

```javascript
fetch('manifest.json').then(r => r.json()).then(m => {
  const b = m.bounds;
  for (const [name, p] of Object.entries(m.parameters)) {
    L.imageOverlay(p.png_path, b, {opacity: 0.7}).addTo(map);
  }
});
```

Works with Leaflet, Mapbox GL, MapLibre GL, or OpenLayers.

## Parameters (46 total)

### Thermodynamic (14)

| Key | Parameter | Units |
|-----|-----------|-------|
| sbcape | Surface-based CAPE | J/kg |
| mlcape | Mixed-layer CAPE (lowest 100 hPa) | J/kg |
| mucape | Most-unstable CAPE | J/kg |
| sb3cape | Surface-based 0-3 km CAPE | J/kg |
| sbcin | Surface-based CIN | J/kg |
| mlcin | Mixed-layer CIN | J/kg |
| mucin | Most-unstable CIN | J/kg |
| dcape | Downdraft CAPE | J/kg |
| lcl_p | Lifted condensation level | hPa |
| theta_e_sfc | Surface equivalent potential temperature | K |
| theta_e_850 | 850 mb equivalent potential temperature | K |
| wetbulb_sfc | Surface wet-bulb temperature | C |
| mixing_ratio | Surface mixing ratio | g/kg |
| pw | Precipitable water | mm |

### Lapse rates (2)

| Key | Parameter | Units |
|-----|-----------|-------|
| lr_700_500 | 700-500 mb lapse rate | C/km |
| lr_850_500 | 850-500 mb lapse rate | C/km |

### Kinematic (7)

| Key | Parameter | Units |
|-----|-----------|-------|
| shear_01km | 0-1 km bulk shear | kt |
| shear_03km | 0-3 km bulk shear | kt |
| shear_06km | 0-6 km bulk shear | kt |
| srh_01km | 0-1 km storm-relative helicity | m^2/s^2 |
| srh_03km | 0-3 km storm-relative helicity | m^2/s^2 |
| bunkers_rm | Bunkers right-mover storm motion | m/s (u, v) |
| bunkers_lm | Bunkers left-mover storm motion | m/s (u, v) |

### Stability indices (5)

| Key | Parameter | Units |
|-----|-----------|-------|
| k_index | K-Index | unitless |
| total_totals | Total Totals Index | unitless |
| lifted_index | Lifted Index | C |
| boyden | Boyden Index | unitless |
| sweat | SWEAT Index | unitless |

### Composite / severe (7)

| Key | Parameter |
|-----|-----------|
| stp | Significant Tornado Parameter |
| scp | Supercell Composite Parameter |
| scp_enhanced | Enhanced SCP (with CIN term) |
| ship | Significant Hail Parameter |
| dcp | Derecho Composite Parameter |
| ehi | Energy-Helicity Index |
| critical_angle | Critical Angle (inflow vs. low-level shear) |

### Surface / heat stress (7)

| Key | Parameter | Units |
|-----|-----------|-------|
| t2m_f | 2-m temperature | F |
| td2m_f | 2-m dewpoint | F |
| heat_index | Heat index (Rothfusz/NWS) | F |
| windchill | Wind chill (NWS) | F |
| wbgt | Wet Bulb Globe Temperature (Dimiceli) | F |

WBGT uses the Dimiceli & Piltz (NWS Tulsa) globe temperature formula with HRRR downward shortwave radiation and solar zenith angle computation. Reference: Dimiceli, V.E., S.F. Piltz, and S.A. Amburn, "Estimation of Black Globe Temperature for Calculation of the WBGT Index."

### Fire weather (4)

| Key | Parameter | Units |
|-----|-----------|-------|
| ffwi | Fosberg Fire Weather Index | unitless |
| hdw | Hot-Dry-Windy Index | unitless |

## Performance

Full HRRR CONUS grid (1,905,141 points, 22 pressure levels):

| Stage | Time |
|-------|------|
| Model ingest | ~80s (network-bound) |
| Obs fetch (ASOS + NDBC) | ~10s |
| 3-pass QC | <1s |
| Barnes analysis (5 fields, 2-pass) | ~28s |
| Parameter computation (46 fields) | ~8s |
| Web export (37 overlays) | ~4.7s (cached) |

RAP is faster due to the smaller grid (~152K points vs 1.9M).

All parameter computation runs through metrust's Rust engine. SBCAPE over 1.9M columns with 22 pressure levels completes in ~400ms. The equivalent pure-Python computation takes over an hour.

## Architecture

```
mesoanalysis/
    pipeline.py          # orchestrator
    config.py            # model configs, Barnes tuning, QC thresholds
    models.py            # ModelData container
    ingest.py            # generic model loader (HRRR, RAP, NAM)
    hrrr/
        ingest.py        # backward-compatible HRRR wrapper
    obs/
        fetch_multi.py   # multi-source obs aggregator
        fetch.py         # ASOS via Iowa Mesonet
        fetch_ndbc.py    # NDBC marine buoys
        fetch_mesonets.py # DCP/RWIS/COOP
        fetch_nws.py     # NWS API
        fetch_cwop.py    # citizen weather (CWOP/APRS)
        qc.py            # 3-pass QC (range + gross-error + buddy)
        models.py        # SurfaceObs / ObsCollection
    analysis/
        barnes.py        # Barnes objective analysis (metrust Rust backend)
        fields.py        # merge corrected fields into model grid
    params/
        thermodynamic.py # CAPE, CIN, LCL, theta-e, lapse rates, PW
        kinematic.py     # bulk shear, SRH, Bunkers storm motion
        composite.py     # STP, SCP, SHIP, DCP, EHI, critical angle
        indices.py       # K-Index, Total Totals, Lifted Index, Boyden, SWEAT
        surface.py       # heat index, wind chill, WBGT, FFWI, HDW
    plotting/
        render.py        # static matplotlib maps
        styles.py        # colormaps and contour levels
    output/
        export.py        # web overlay export (PNG + manifest + grids)
web/
    index.html           # test viewer (Leaflet dark theme)
    app.js               # frontend logic
    style.css            # dark theme
    serve.py             # development HTTP server with point query API
```

## Dependencies

- [metrust](https://github.com/FahrenheitResearch/metrust-py) -- Rust-based meteorological computation engine (CAPE, SRH, shear, SHIP, DCP, WBGT, and 200+ functions)
- [Herbie](https://herbie.readthedocs.io/) -- GRIB2 data access
- NumPy, SciPy, Cartopy, Matplotlib, Pillow, requests, pandas

## License

MIT. See [LICENSE](LICENSE).
