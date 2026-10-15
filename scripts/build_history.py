"""Plan and execute bounded, resumable Sentinel-2 monthly acquisition history."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forestmind.data.history import HistoryRequest, create_history_plan, execute_history_plan, load_history_plan
from forestmind.monitoring.ingest import prepare_cached_observations, validate_bbox, write_prepared_import


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Freeze a bounded monthly Sentinel-2 candidate plan, then acquire at most one quality-passing "
        "scene per month. No interpolation, automatic provider fallback, training labels or forecasts."
    ))
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Query monthly windows and save unsigned STAC candidates")
    planning.add_argument("--plan", required=True, help="New frozen plan JSON; existing files are refused")
    planning.add_argument("--provider", required=True, choices=["earth-search", "planetary-computer"])
    planning.add_argument("--region", required=True, help="--region=west,south,east,north")
    planning.add_argument("--start", required=True, help="Inclusive YYYY-MM-DD, UTC")
    planning.add_argument("--end", required=True, help="Inclusive YYYY-MM-DD, UTC; at most 120 months")
    planning.add_argument("--patch-size", type=int, default=64)
    planning.add_argument("--max-cloud-cover", type=float, default=30.)
    planning.add_argument("--min-valid-fraction", type=float, default=.6, help="Required AOI index-valid pixel fraction")
    planning.add_argument("--candidates-per-month", type=int, default=3, help="Rank and freeze at most 20")
    planning.add_argument("--search-limit", type=int, default=30, help="Bounded catalog subset per month (maximum 100)")
    planning.add_argument("--max-attempts", type=int, default=12, help="Candidate attempts per execute invocation, including cache reads")
    planning.add_argument("--max-downloads", type=int, default=6, help="Network candidate attempts per execute invocation; failures count")
    planning.add_argument("--timeout", type=float, default=30., help="Finite timeout per provider/raster request, in seconds")
    execution = commands.add_parser("execute", help="Execute or resume a frozen plan with its saved bounds")
    execution.add_argument("--plan", required=True)
    execution.add_argument("--report", required=True, help="Progress JSON, atomically updated after every attempt")
    execution.add_argument("--cache-dir", required=True)
    execution.add_argument("--resume", action="store_true", help="Continue remaining candidates; revalidate every prior success")
    execution.add_argument("--output", help="Optional new import JSON plus provenance, selecting accepted scenes only")
    execution.add_argument("--name", help="Monitoring region name, required with --output")
    execution.add_argument("--country", default="Unspecified")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            request = HistoryRequest(provider=args.provider, bbox=validate_bbox(args.region.split(",")),
                                     start=args.start, end=args.end, patch_size=args.patch_size,
                                     max_cloud_cover=args.max_cloud_cover, min_valid_fraction=args.min_valid_fraction,
                                     candidates_per_month=args.candidates_per_month, search_limit=args.search_limit,
                                     max_attempts=args.max_attempts, max_downloads=args.max_downloads, timeout_seconds=args.timeout)
            plan = create_history_plan(request, args.plan)
            print(f"Saved {len(plan['months'])} monthly candidate windows: {Path(args.plan).resolve()}")
            failed = sum(row["status"] == "search_failed" for row in plan["months"])
            print("Candidate ranking is limited to each provider's bounded returned subset.")
            if failed:
                print(f"{failed} catalog queries failed; their reasons are preserved in the plan.")
                return 2
            return 0
        if args.output and not (args.name and args.name.strip()):
            raise ValueError("--name is required with --output")
        if args.name and not args.output:
            raise ValueError("--name requires --output")
        if args.output:
            if not 2 <= len(args.name) <= 120 or not 2 <= len(args.country) <= 100:
                raise ValueError("Import name must have 2-120 characters and country 2-100 characters")
            output = Path(args.output).resolve()
            sidecar = output.with_suffix(".provenance.json")
            if output.suffix.lower() != ".json" or output == sidecar:
                raise ValueError("Choose an import .json name that does not end in .provenance.json")
            protected = {Path(args.plan).resolve(), Path(args.report).resolve()}
            if output in protected or sidecar in protected or output.exists() or sidecar.exists():
                raise ValueError("Import output/provenance must be new paths distinct from the plan and report")
        report = execute_history_plan(args.plan, args.report, args.cache_dir, resume=args.resume)
        accepted = report["accepted_cache_paths"]
        print(f"Accepted {len(accepted)} / {len(report['months'])} months; status: {report['status']}")
        print(f"Progress and rejection reasons: {Path(args.report).resolve()}")
        if args.output and accepted:
            _, request = load_history_plan(args.plan)
            prepared = prepare_cached_observations(args.cache_dir, sensor="sentinel2", name=args.name,
                                                  country=args.country, bbox=request.bbox, source_paths=accepted)
            prepared.provenance["history_selection"] = {
                "plan_sha256": report["plan_sha256"], "report": str(Path(args.report).resolve()),
                "report_sha256": hashlib.sha256(Path(args.report).read_bytes()).hexdigest(),
                "status": report["status"], "requested_months": len(report["months"]),
                "accepted_months": len(accepted), "min_valid_fraction": request.min_valid_fraction,
                "missing_months": [row["month"] for row in report["months"] if not row["accepted"]],
            }
            output, sidecar = write_prepared_import(prepared, args.output)
            print(f"Accepted-only observation import: {output}")
            print(f"Acquisition provenance: {sidecar}")
        elif args.output:
            print("No usable observations; no import JSON was fabricated.")
        return 0 if report["status"] == "complete" else 2
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f"Cannot build history: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
