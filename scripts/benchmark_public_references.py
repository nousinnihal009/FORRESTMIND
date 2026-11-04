"""Freeze or run a separate retrospective experiment on public point references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forestmind.data.public_forest_reference import load_public_reference
from forestmind.training.public_reference_benchmark import freeze_public_protocol, run_public_benchmark, write_new_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run" and args.output is None:
            raise ValueError("run requires a new --output path")
        destination = args.protocol if args.command == "freeze" else args.output
        if destination.exists():
            raise ValueError("Output already exists; frozen protocols/results cannot be overwritten")
        packet = load_public_reference(args.packet)
        if args.command == "freeze":
            result = freeze_public_protocol(packet)
        else:
            if args.protocol.stat().st_size > 16 * 1024**2:
                raise ValueError("Protocol exceeds 16 MiB")
            protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
            result = run_public_benchmark(packet, protocol)
        write_new_record(result, destination)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Public reference experiment unavailable: {error}", file=sys.stderr)
        return 2
    print(f"Saved: {destination.resolve()}")
    print("Retrospective point-reference study; no operational forest prediction claim.")
    print(json.dumps(result["audit"]["split_counts"], indent=2))
    if args.command == "run":
        print(json.dumps({name: metrics["test"] for name, metrics in result["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
