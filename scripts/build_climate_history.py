"""Acquire a bounded real ERA5 history and freeze separate monthly diagnostics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forestmind.data.climate_history import build_climate_history
from forestmind.monitoring.climatology import read_json_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, type=Path, help="Saved area JSON containing a bbox")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory, or existing directory with --resume")
    parser.add_argument("--baseline-start", default="1991-01-01")
    parser.add_argument("--baseline-end", default="2020-12-31")
    parser.add_argument("--observed-start", default="2021-01-01")
    parser.add_argument("--observed-end", default="2025-12-31")
    parser.add_argument("--max-requests", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        area, _ = read_json_record(args.area)
        result = build_climate_history(area["bbox"], args.output,
                    baseline_start=args.baseline_start, baseline_end=args.baseline_end,
                    observed_start=args.observed_start, observed_end=args.observed_end,
                    max_requests=args.max_requests, timeout=args.timeout, resume=args.resume)
        print(json.dumps({key: result[key] for key in (
            "status", "accepted_windows", "requests_this_invocation",
            "available_reference_feature_months", "observed_months", "available_observed_feature_months") if key in result}, indent=2))
        print(f"Evidence and resumable progress: {args.output.resolve()}")
        print("Retrospective ERA5 diagnostics only; no validated drought, forest outcome, or historical-availability claim.")
        return 0 if result["status"] == "complete" else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Climate history unavailable: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
