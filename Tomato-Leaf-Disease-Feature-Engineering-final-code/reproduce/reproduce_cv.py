# -*- coding: utf-8 -*-
"""Reproduce the reviewer-oriented five-fold validation from cached training features."""

import argparse
import json
from pathlib import Path

import numpy as np

from cross_validation.five_fold_cv import run_five_fold_cv
from feature_extraction.feature_names import build_feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-train",
        default="outputs/features/X_train_raw.npy",
    )
    parser.add_argument(
        "--y-train",
        default="outputs/features/y_train.npy",
    )
    parser.add_argument(
        "--class-map",
        default=None,
        help="Optional JSON containing {'class_to_id': {...}}.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/five_fold_cv",
    )
    args = parser.parse_args()

    X = np.load(args.x_train)
    y = np.load(args.y_train)

    feature_names = build_feature_names()

    id_to_class = None
    if args.class_map is not None:
        with Path(args.class_map).open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = json.load(f)
        class_to_id = payload["class_to_id"]
        id_to_class = {
            int(v): k
            for k, v in class_to_id.items()
        }

    results = run_five_fold_cv(
        X=X,
        y=y,
        feature_names=feature_names,
        output_dir=args.output_dir,
        id_to_class=id_to_class,
    )

    print(results["summary"])
    print(results["feature_count_summary"])


if __name__ == "__main__":
    main()
