"""Freeze local ERA5 references or calculate auditable monthly departures.

No downloads, model training, database changes, or existing-file replacement.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forestmind.monitoring.climatology import (
    analyze_climate, artifact_hash, freeze_climatology,
    read_json_record, write_new_artifact,
)


def _read_contexts(paths):
    contexts, files = [], []
    for path in paths:
        value, provenance = read_json_record(path)
        # A selected-area export is also an existing evidence source. Keep the
        # full input-file hash, even when only its saved climate field is used.
        if value.get("kind") == "forestmind_selected_area_analysis":
            value = value.get("climate")
        if not isinstance(value, dict):
            raise ValueError("Input must be a saved climate context or a selected-area export containing one")
        contexts.append(value)
        files.append(provenance)
    return contexts, files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Freeze a 30–60 calendar-year reference from saved daily ERA5 contexts")
    freeze.add_argument("--baseline-start", required=True, help="Inclusive YYYY-01-01 reference start")
    freeze.add_argument("--baseline-end", required=True, help="Inclusive YYYY-12-31 reference end")
    analyze = commands.add_parser("analyze", help="Apply a frozen reference to separate observed monthly evidence")
    analyze.add_argument("--reference", required=True, type=Path)
    analyze.add_argument("--cutoff", required=True, help="Latest permitted observation date, YYYY-MM-DD")
    for command in (freeze, analyze):
        command.add_argument("--context", required=True, type=Path, action="append",
                             help="Saved ERA5 context or selected-area export; repeat for disjoint source windows (maximum 12)")
        command.add_argument("--output", required=True, type=Path, help="New JSON artifact path; never overwritten")
    args = parser.parse_args(argv)
    try:
        if len(args.context) > 12:
            raise ValueError("At most 12 bounded context files are supported")
        if args.output.exists():
            raise ValueError("Output already exists; choose a new artifact path")
        sources = args.context + ([args.reference] if args.command == "analyze" else [])
        if any(args.output.resolve() == path.resolve() for path in sources):
            raise ValueError("Output must differ from every input; source evidence must be preserved")
        contexts, input_files = _read_contexts(args.context)
        if args.command == "freeze":
            result = freeze_climatology(contexts, baseline_start=args.baseline_start, baseline_end=args.baseline_end)
        else:
            reference, reference_file = read_json_record(args.reference)
            result = analyze_climate(contexts, reference, cutoff=args.cutoff)
            result["reference_file"] = {**reference_file, "artifact_sha256": reference["artifact_sha256"]}
        result["input_files"] = input_files
        result["artifact_sha256"] = artifact_hash(result)
        write_new_artifact(result, args.output)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"Climate diagnostics unavailable: {error}", file=sys.stderr)
        return 2
    print(f"Saved climate evidence: {args.output.resolve()}")
    if args.command == "freeze":
        print(f"Eligible reference feature/month pairs: {result['available_feature_months']} / 48")
    else:
        available = sum(value["available"] for month in result["months"] for value in month["variables"].values())
        print(f"Available feature/month departures: {available} / {len(result['months']) * 4}")
    print("Descriptive reference departures only; no validated drought diagnosis, ecological prediction or model training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
