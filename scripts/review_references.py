"""Create, check or prepare reviewed annual tree-cover-loss reference data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forestmind.data.reference_review import reference_template, review_references, prepare_reviewed_references
from forestmind.training.reference_experiment import write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output", required=True)
    check = commands.add_parser("check")
    check.add_argument("--manifest", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            write_new_json(reference_template(), args.output)
            print("Created empty reference intake. No labels or predictions were generated.")
            return 0
        result = (review_references(args.manifest)[0] if args.command == "check"
                  else prepare_reviewed_references(args.manifest, args.output))
        print(json.dumps({key: result[key] for key in ("counts", "class_counts", "ready_for_baseline_experiment", "blockers")}, indent=2))
        return 0 if result["ready_for_baseline_experiment"] else 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("Reference intake refused:", error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
