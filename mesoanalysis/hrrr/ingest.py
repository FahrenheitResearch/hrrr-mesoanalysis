"""HRRR data ingest — backward-compatible wrapper.

The implementation has moved to ``mesoanalysis.ingest.load_model``.
This module re-exports ``HRRRData`` and ``load_hrrr`` so that existing
code using ``from mesoanalysis.hrrr.ingest import HRRRData, load_hrrr``
continues to work unchanged.
"""

from __future__ import annotations

from datetime import datetime

from mesoanalysis.models import ModelData

# Backward-compatible alias
HRRRData = ModelData


def load_hrrr(dt: datetime, fxx: int = 0) -> ModelData:
    """Fetch HRRR surface + pressure-level data and return `ModelData`.

    This is a thin wrapper around :func:`mesoanalysis.ingest.load_model`
    with ``model="hrrr"``.

    Parameters
    ----------
    dt : datetime
        Model initialisation time (UTC).
    fxx : int
        Forecast hour (0 for analysis).

    Returns
    -------
    ModelData
    """
    from mesoanalysis.ingest import load_model

    return load_model(dt, model="hrrr", fxx=fxx)
