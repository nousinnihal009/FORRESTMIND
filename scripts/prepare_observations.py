"""Convert a local optical adapter cache into a strict dashboard import JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forestmind.monitoring.ingest import prepare_cached_observations, validate_bbox, write_prepared_import


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Aggregate locally cached, quality-masked optical acquisitions into monthly ForestMind "
        "observations. No network request or model inference is performed."
    ))
    parser.add_argument("--cache-dir", required=True, help="Adapter cache root or optical sensor directory")
    parser.add_argument("--sensor", choices=["sentinel2", "s2", "landsat", "landsat8", "landsat9"], required=True)
    parser.add_argument("--name", required=True, help="Monitoring region name")
    parser.add_argument("--country", default="Unspecified")
    parser.add_argument("--region", help="Select an exact cached bbox using --region=west,south,east,north")
    parser.add_argument("--output", required=True, help="New import JSON path; also writes .provenance.json")
    args = parser.parse_args(argv)
    try:
        bbox = validate_bbox(args.region.split(",")) if args.region else None
        prepared = prepare_cached_observations(args.cache_dir, sensor=args.sensor, name=args.name,
                                                country=args.country, bbox=bbox)
        output, sidecar = write_prepared_import(prepared, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Cannot prepare observations: {error}\n")
    print(f"Prepared {len(prepared.payload['observations'])} observed months: {output}")
    print(f"Acquisition provenance: {sidecar}")
    print("Import the main JSON through the dashboard's Import observations action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
