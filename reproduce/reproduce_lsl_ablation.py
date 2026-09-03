# -*- coding: utf-8 -*-
"""
Dedicated Full-vs-w/o-LSL ablation.

Full     = 296 dimensions.
w/o LSL  = 294 dimensions, removing only:
    lesion_edge_ratio
    lesion_inner_ratio

No permutation-importance pruning, Pearson pruning, feature-group weighting,
or feature-space augmentation is applied.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classification.evaluate_svm import evaluate_model
from feature_extraction.feature_names import build_feature_names
from utils.class_mapping import load_class_mapping
from utils.dataset_loader import load_split_features


RANDOM_STATE = 42

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 5.0,
    "gamma": "scale",
    "class_weight": "balanced",
}

FEATURE_CONFIG = dict(
    use_color=True,
    use_lbp=True,
    use_shape=True,
    use_lesion=True,
)

LSL_NAMES = {
    "lesion_edge_ratio",
    "lesion_inner_ratio",
}


def build_svm():
    return SVC(
        **SVM_PARAMS,
        probability=False,
        random_state=RANDOM_STATE,
    )


def train_and_evaluate(
    name,
    X_train,
    y_train,
    X_test,
    y_test,
    class_to_id,
    output_dir,
):
    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(
        X_train
    )
    X_test_s = scaler.transform(
        X_test
    )

    model = build_svm()

    model.fit(
        X_train_s,
        y_train,
    )

    _, test_metrics, _ = evaluate_model(
        model,
        X_test_s,
        y_test,
        class_to_id,
        f"Test_{name}",
        output_dir,
    )

    return {
        "setting": name,
        "n_features": X_train.shape[1],
        **test_metrics,
    }


def run_lsl_ablation(
    data_root,
    class_map_path,
    mask_root,
    output_root,
):
    data_root = Path(data_root)

    (
        class_to_id,
        _,
        _,
    ) = load_class_mapping(
        class_map_path
    )

    X_train, y_train = load_split_features(
        data_root / "train",
        class_to_id=class_to_id,
        mask_root=mask_root,
        common_root=data_root,
        **FEATURE_CONFIG,
    )

    X_test, y_test = load_split_features(
        data_root / "test",
        class_to_id=class_to_id,
        mask_root=mask_root,
        common_root=data_root,
        **FEATURE_CONFIG,
    )

    names = build_feature_names(
        **FEATURE_CONFIG
    )

    keep_idx = [
        i
        for i, name in enumerate(names)
        if name not in LSL_NAMES
    ]

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    output_dir = (
        Path(output_root)
        / timestamp
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    rows.append(
        train_and_evaluate(
            "Full",
            X_train,
            y_train,
            X_test,
            y_test,
            class_to_id,
            output_dir,
        )
    )

    rows.append(
        train_and_evaluate(
            "wo_LSL",
            X_train[:, keep_idx],
            y_train,
            X_test[:, keep_idx],
            y_test,
            class_to_id,
            output_dir,
        )
    )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        output_dir / "lsl_ablation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--class-map",
        default="datasets/class_to_id.json",
    )
    parser.add_argument(
        "--mask-root",
        default="masks",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/ablation/lsl",
    )
    args = parser.parse_args()

    run_lsl_ablation(
        data_root=args.data_root,
        class_map_path=args.class_map,
        mask_root=args.mask_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
