"""Prepare a dated Sentinel-2 inference NPZ without inventing labels or modalities."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bbox = [float(value) for value in args.region.split(",")] if args.region else None
        prepared = prepare_sequence(args.cache_dir, through=args.through, time_steps=args.time_steps,
                                    bbox=bbox, min_valid_fraction=args.min_valid_fraction)
        output, provenance = write_sequence(prepared, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Cannot prepare inference sequence: {error}\n")
    print(f"Saved {prepared.provenance['selected_acquisitions']} actual acquisitions: {output}")
    print(f"Observed through: {prepared.provenance['observed_through']}")
    print(f"Provenance: {provenance}")
    print("Raw reflectance; SAR absent and masked. No supervised labels. Match checkpoint preprocessing before inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
