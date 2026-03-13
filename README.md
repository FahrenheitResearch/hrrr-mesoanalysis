# hrrr-mesoanalysis

SPC-style mesoscale analysis at 3km HRRR resolution. Blends real ASOS surface obs into the HRRR grid via Barnes objective analysis, then runs a full severe weather parameter suite using [metrust](https://github.com/FahrenheitResearch/metrust-py).

```bash
pip install metrust herbie-data cartopy scipy matplotlib requests
python -m mesoanalysis 2025-05-10T18:00
```

PNGs go to `output/YYYYMMDD_HHMM/`.

## How it works

Fetches HRRR surface + 22 pressure levels through [Herbie](https://github.com/blaylockbk/Herbie), pulls ~1,700 ASOS obs from the Iowa Environmental Mesonet, QC's them against the HRRR first guess, then runs a multi-pass Barnes analysis to nudge the model grid toward observed values. The corrected fields feed into metrust for parameter computation across the full 1799x1059 HRRR grid.

The [SPC Mesoanalysis](https://www.spc.noaa.gov/exper/mesoanalysis/) does roughly the same thing but on the 13km RAP. This runs on the 3km HRRR — about 19x more grid columns — which is only feasible because metrust handles the compute. SBCAPE over 1.9M columns with 22 levels runs in ~400ms. With MetPy that's closer to an hour.

## Parameters

**Thermodynamic** — SBCAPE, SBCIN, MLCAPE, MUCAPE, SB3CAPE, LCL, theta-e (sfc + 850mb), wet bulb, 700-500mb lapse rate, mixing ratio, precipitable water

**Kinematic** — 0-1km, 0-3km, 0-6km bulk shear; 0-1km and 0-3km SRH

**Composite** — STP, SCP

**Surface** — 2m temp, 2m dewpoint

## Config

`mesoanalysis/config.py` has the Barnes tuning (kappa, gamma, passes), domain extent, obs time window, and output settings.

## License

MIT
