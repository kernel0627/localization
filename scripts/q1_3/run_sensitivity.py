"""Run main's finite gain and initial-perturbation checks."""

from pathlib import Path

from scripts.q1_3.run_iterative_reference_sensitivity import main as run_checks


def main():
    run_checks(Path(__file__).resolve().parents[2] / "outputs/q1_3")


if __name__ == "__main__":
    main()
