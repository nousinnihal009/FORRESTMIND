"""Infer from one explicit NPZ sequence and save a portable, uncalibrated report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forestmind.inference.predict import predict_checkpoint, write_inference_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local checkpoint; safe weights-only loading")
    parser.add_argument("--input", type=Path, required=True, help="Raw NPZ sequence with explicit masks and timestamps")
    parser.add_argument("--region-id", required=True, help="Existing monitoring region ID used for report association")
    parser.add_argument("--observed-through", required=True, help="Latest actual acquisition date, YYYY-MM-DD")
    parser.add_argument("--synthetic-input", action="store_true", help="Explicitly mark artificial inputs as a demonstration")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON for the dashboard's Model runs importer")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.output.resolve() in {args.input.resolve(), args.checkpoint.resolve()}:
            raise ValueError("Output must not overwrite the checkpoint or input sequence")
        report = predict_checkpoint(args.checkpoint, args.input, region_id=args.region_id,
                                    observed_through=args.observed_through,
                                    synthetic_input=args.synthetic_input, device=args.device)
        output = write_inference_report(report, args.output, protected_paths=(args.checkpoint, args.input))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        logging.error("Inference failed: %s", exc)
        return 2
    print(f"Saved uncalibrated model report: {output}")
    print(f"Input synthetic: {report['input']['synthetic']}; checkpoint trained on synthetic: {report['checkpoint']['trained_on_synthetic']}")
    print("Not a validated ecological assessment. Import this JSON under the matching region's Model runs tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
