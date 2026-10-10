"""
Generate synthetic satellite data for ForestMind training.

Usage:
    python scripts/generate_synthetic_data.py [--num-train 2000] [--num-val 400] [--output-dir data/]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forestmind.data.synthetic import SyntheticForestGenerator, HEALTH_STATES, DRIVER_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ForestMind data")
    parser.add_argument("--num-train", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=400, help="Number of validation samples")
    parser.add_argument("--num-test", type=int, default=200, help="Number of test samples")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size in pixels")
    parser.add_argument("--time-steps", type=int, default=12, help="Number of time steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    splits = {
        "train": (args.num_train, args.seed),
        "val": (args.num_val, args.seed + 10000),
        "test": (args.num_test, args.seed + 20000),
    }

    for split_name, (num_samples, seed) in splits.items():
        print(f"\n{'=' * 50}")
        print(f"  Generating {split_name} split: {num_samples} samples")
        print(f"{'=' * 50}")

        split_dir = output_dir / split_name
        generator = SyntheticForestGenerator(
            patch_size=args.patch_size,
            time_steps=args.time_steps,
            seed=seed,
        )

        start_time = time.time()
        samples = generator.generate_dataset(num_samples, output_dir=split_dir)
        elapsed = time.time() - start_time

        # Statistics
        health_counts = {state: 0 for state in HEALTH_STATES}
        driver_counts = {driver: 0 for driver in DRIVER_CLASSES}
        scores = []

        for sample in samples:
            state_idx = sample["health_state"]
            health_counts[HEALTH_STATES[state_idx]] += 1
            scores.append(sample["health_score"])
            for i, driver in enumerate(DRIVER_CLASSES):
                if sample["drivers"][i] > 0:
                    driver_counts[driver] += 1

        print(f"\n  Generated in {elapsed:.1f}s ({num_samples / elapsed:.1f} samples/s)")
        print(f"  Saved to: {split_dir.resolve()}")
        print(f"\n  Health distribution:")
        for state, count in health_counts.items():
            pct = count / num_samples * 100
            bar = "█" * int(pct / 5)
            print(f"    {state:20s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n  Average health score: {sum(scores) / len(scores):.1f}")
        print(f"\n  Driver frequency:")
        for driver, count in sorted(driver_counts.items(), key=lambda x: -x[1]):
            pct = count / num_samples * 100
            print(f"    {driver:20s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n{'=' * 50}")
    print(f"  ✓ Dataset generation complete")
    print(f"  Total: {sum(n for n, _ in splits.values())} samples")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
