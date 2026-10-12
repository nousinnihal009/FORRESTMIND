"""Backtest monitoring baselines on a prepared import JSON, with auditable cutoffs.

Example: python scripts/backtest_monitoring.py --input prepared.json --output backtest.json
Requires the same optional web dependencies as the prepared observation importer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forestmind.monitoring.evaluation import backtest_prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin evaluation of transparent monitoring baselines")
    parser.add_argument("--input", required=True, type=Path, help="Prepared import JSON used by the monitoring application")
    parser.add_argument("--output", type=Path, help="Write JSON results here; otherwise emit JSON to stdout")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 6, 12], choices=[1, 6, 12],
                        help="Increasing calendar-month horizons (default: 1 6 12)")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input; source observations must be preserved")
    try:
        raw = args.input.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
        result = backtest_prepared(payload, horizons_months=args.horizons)
        result["provenance"]["input_file"] = str(args.input.resolve())
        result["provenance"]["input_file_sha256"] = hashlib.sha256(raw).hexdigest()
        rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Saved evaluation: {args.output.resolve()}", file=sys.stderr)
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(rendered)
        if result["synthetic_warning"]:
            print(result["synthetic_warning"], file=sys.stderr)
        for method, scores in result["methods"].items():
            values = scores["aggregate"]
            print(f"{method}: n={values['n']}, MAE={values['mae']}, RMSE={values['rmse']}", file=sys.stderr)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
