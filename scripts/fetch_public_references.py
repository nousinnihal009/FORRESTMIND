"""Fetch or audit the pinned, published global forest interpretation reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forestmind.data.public_forest_reference import fetch_public_references, load_public_reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New packet directory, or matching packet to resume/verify")
    parser.add_argument("--check", action="store_true", help="Audit existing local evidence only; no network")
    args = parser.parse_args(argv)
    try:
        packet = load_public_reference(args.output) if args.check else fetch_public_references(args.output)
    except (ValueError, OSError, TypeError) as error:
        print(f"Public reference unavailable: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset_id": packet["metadata"]["dataset_id"],
                      "packet": str(args.output.resolve()), "audit": packet["audit"]}, indent=2))
    print("Published retrospective interpretations; separate from the 100 ha loss target and operational prediction validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
