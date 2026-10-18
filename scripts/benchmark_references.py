"""Freeze a real-reference protocol, then evaluate fixed conventional baselines."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forestmind.training.reference_experiment import freeze_protocol, run_reference_experiment, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--input", required=True)
    freeze.add_argument("--protocol", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if args.command == "freeze":
            result = freeze_protocol(data)
            write_new_json(result, args.protocol)
            print("Frozen reference protocol:", result["protocol_sha256"])
        else:
            protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
            result = run_reference_experiment(data, protocol)
            write_new_json(result, args.output)
            print("Saved reference comparison:", args.output)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("Reference experiment refused:", error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
