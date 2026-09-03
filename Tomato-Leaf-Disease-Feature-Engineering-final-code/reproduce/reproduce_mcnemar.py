# -*- coding: utf-8 -*-
"""Reproduce the independent-test Full-vs-Pruned McNemar analysis."""

import argparse

from statistical_analysis.mcnemar_test import (
    run_mcnemar_from_saved_models,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        required=True,
    )
    parser.add_argument(
        "--x-test",
        required=True,
    )
    parser.add_argument(
        "--y-test",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="results/mcnemar",
    )
    args = parser.parse_args()

    result = run_mcnemar_from_saved_models(
        model_file=args.model_file,
        X_test_file=args.x_test,
        y_test_file=args.y_test,
        save_dir=args.output_dir,
        alpha=0.05,
    )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
