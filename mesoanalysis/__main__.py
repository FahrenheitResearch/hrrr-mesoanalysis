"""CLI entry point: python -m mesoanalysis 2026-03-10T18:00 --model hrrr --fxx 0"""
import argparse
from datetime import datetime

from .pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Run mesoscale analysis for a given time and NWP model.",
    )
    parser.add_argument(
        "analysis_time",
        nargs="?",
        default="2026-03-10T18:00",
        help="Analysis time in ISO format (default: 2026-03-10T18:00)",
    )
    parser.add_argument(
        "--model",
        default="hrrr",
        choices=["hrrr", "rap", "nam"],
        help="NWP model to use (default: hrrr)",
    )
    parser.add_argument(
        "--fxx",
        type=int,
        default=0,
        help="Forecast hour (default: 0 for analysis)",
    )

    args = parser.parse_args()
    dt = datetime.fromisoformat(args.analysis_time)
    run(dt, model=args.model, fxx=args.fxx)


if __name__ == "__main__":
    main()
