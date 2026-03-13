"""CLI entry point: python -m mesoanalysis 2026-03-10T18:00"""
import sys
from datetime import datetime
from .pipeline import run

if __name__ == "__main__":
    dt = (
        datetime.fromisoformat(sys.argv[1])
        if len(sys.argv) > 1
        else datetime(2026, 3, 10, 18, 0)
    )
    run(dt)
