"""Prepare a dated Sentinel-2 inference NPZ without inventing labels or modalities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forestmind.data.sequence import prepare_sequence, write_sequence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--through", required=True, help="Latest allowed UTC acquisition date, YYYY-MM-DD")
    parser.add_argument("--time-steps", type=int, default=12, help="Maximum recent eligible acquisitions; no padding")
    parser.add_argument("--region", help="Optional west,south,east,north to choose one AOI")
    parser.add_argument("--min-valid-fraction", type=float, default=0.4)
    parser.add_argument("--radar-context", type=Path, help="Saved Sentinel-1 RTC context JSON on the same AOI/grid")
    parser.add_argument("--climate-context", type=Path, help="Saved ERA5 context JSON with daily raw_response")
    parser.add_argument("--radar-max-age-days", type=float, default=12)
    parser.add_argument("--radar-min-valid-fraction", type=float, default=.4)
    parser.add_argument("--climate-window-days", type=int, default=30)
    parser.add_argument("--climate-lag-days", type=int, default=5,
                        help="Exclude recent daily records; does not verify historical reanalysis vintage")
    parser.add_argument("--climate-min-coverage", type=float, default=1.0,
                        help="Valid days / window days per feature; partial sums are never extrapolated")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        def read_context(path):
            if path is None:
                return None
            if path.stat().st_size > 16 * 1024**2:
                raise ValueError("Context JSON exceeds the 16 MiB limit")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Context JSON must contain one object")
            return value

        bbox = [float(value) for value in args.region.split(",")] if args.region else None
        prepared = prepare_sequence(args.cache_dir, through=args.through, time_steps=args.time_steps,
            bbox=bbox, min_valid_fraction=args.min_valid_fraction,
            radar_context=read_context(args.radar_context), climate_context=read_context(args.climate_context),
            radar_max_age_days=args.radar_max_age_days, radar_min_valid_fraction=args.radar_min_valid_fraction,
            climate_window_days=args.climate_window_days, climate_lag_days=args.climate_lag_days,
            climate_min_coverage=args.climate_min_coverage)
        output, provenance = write_sequence(prepared, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Cannot prepare inference sequence: {error}\n")
    print(f"Saved {prepared.provenance['selected_acquisitions']} actual acquisitions: {output}")
    print(f"Observed through: {prepared.provenance['observed_through']}")
    print(f"Provenance: {provenance}")
    print("Available measured modalities: " + ", ".join(prepared.provenance["available_modalities"]))
    print("Raw physical units with explicit missingness masks; no supervised labels. Match checkpoint preprocessing before inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
