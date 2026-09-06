"""Run main's receiver-local iterative adjustment on Table 1."""

import argparse
from pathlib import Path

from scripts.q1_3.local_adjustment import LocalSettings
from scripts.q1_3.run_iterative_reference_baseline import (
    run_validation as run_validation,
    simulate_adjustment as simulate_adjustment,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/q1_3")
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()
    run_validation(
        args.output_dir,
        settings=LocalSettings(gain=args.gain),
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
