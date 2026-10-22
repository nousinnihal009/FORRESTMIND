"""Audit independent references and prepare a real, partially labeled dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forestmind.data.training_data import audit_training_manifest, prepare_training_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Versioned training-reference JSON")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Audit only; do not write a dataset")
    action.add_argument("--output", help="New directory for train/val/test files, report and config")
    arguments = parser.parse_args()
    try:
        if arguments.check:
            _, _, report = audit_training_manifest(arguments.manifest)
        else:
            report = prepare_training_dataset(arguments.manifest, arguments.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Training data rejected: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
