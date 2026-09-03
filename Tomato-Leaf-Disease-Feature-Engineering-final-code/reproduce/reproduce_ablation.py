# -*- coding: utf-8 -*-
"""
Reproduce handcrafted-feature ablation experiments.

The original feature-group ablations are implemented through the same feature
extraction switches used by the pipeline:

    use_color
    use_lbp
    use_shape
    use_lesion

Experiments
-----------
Full
w/o Color
w/o LBP
w/o Shape
w/o Lesion

An optional "w/o LSL" experiment is also supported. Because LSL is a
two-dimensional subcomponent of the 22-D lesion group rather than a top-level
feature switch, this experiment extracts the Full feature vector and removes
only:

    lesion_edge_ratio
    lesion_inner_ratio

No feature-group weighting is used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


ABLATION_CONFIGS = {
    "Full": {
        "use_color": True,
        "use_lbp": True,
        "use_shape": True,
        "use_lesion": True,
    },

    "w/o Color": {
        "use_color": False,
        "use_lbp": True,
        "use_shape": True,
        "use_lesion": True,
    },

    "w/o LBP": {
        "use_color": True,
        "use_lbp": False,
        "use_shape": True,
        "use_lesion": True,
    },

    "w/o Shape": {
        "use_color": True,
        "use_lbp": True,
        "use_shape": False,
        "use_lesion": True,
    },

    "w/o Lesion": {
        "use_color": True,
        "use_lbp": True,
        "use_shape": True,
        "use_lesion": False,
    },
}


def build_svm():
    """Fixed RBF-SVM used for all ablation configurations."""
    return SVC(
        kernel=SVM_PARAMS["kernel"],
        C=SVM_PARAMS["C"],
        gamma=SVM_PARAMS["gamma"],
        class_weight=SVM_PARAMS[
            "class_weight"
        ],
        probability=False,
        random_state=RANDOM_STATE,
    )


def evaluate_predictions(
    y_true,
    y_pred,
):
    """Return Accuracy and macro P/R/F1."""
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def cache_paths(
    cache_root,
    experiment_name,
):
    """Create experiment-specific cache paths."""
    safe_name = (
        experiment_name
        .lower()
        .replace(" ", "_")
        .replace("/", "")
    )

    root = (
        Path(cache_root)
        / safe_name
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "x_train": root / "X_train.npy",
        "y_train": root / "y_train.npy",
        "x_test": root / "X_test.npy",
        "y_test": root / "y_test.npy",
    }


def load_or_extract_ablation_features(
    experiment_name,
    feature_kwargs,
    data_root,
    class_to_id,
    mask_root,
    cache_root,
):
    """
    Extract one ablation configuration using the actual feature switches.
    """
    data_root = Path(data_root)

    paths = cache_paths(
        cache_root,
        experiment_name,
    )

    if all(p.exists() for p in paths.values()):
        X_train = np.load(
            paths["x_train"]
        )
        y_train = np.load(
            paths["y_train"]
        )
        X_test = np.load(
            paths["x_test"]
        )
        y_test = np.load(
            paths["y_test"]
        )

    else:
        X_train, y_train = load_split_features(
            data_root / "train",
            class_to_id=class_to_id,
            mask_root=mask_root,
            common_root=data_root,
            **feature_kwargs,
        )

        X_test, y_test = load_split_features(
            data_root / "test",
            class_to_id=class_to_id,
            mask_root=mask_root,
            common_root=data_root,
            **feature_kwargs,
        )

        np.save(
            paths["x_train"],
            X_train,
        )
        np.save(
            paths["y_train"],
            y_train,
        )
        np.save(
            paths["x_test"],
            X_test,
        )
        np.save(
            paths["y_test"],
            y_test,
        )

    feature_names = build_feature_names(
        **feature_kwargs
    )

    if X_train.shape[1] != len(feature_names):
        raise RuntimeError(
            f"{experiment_name}: "
            f"feature dimension mismatch "
            f"{X_train.shape[1]} != "
            f"{len(feature_names)}"
        )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
    )


def remove_lsl_columns(
    X_train,
    X_test,
    feature_names,
):
    """
    Remove only the two LSL dimensions:
    lesion_edge_ratio and lesion_inner_ratio.
    """
    lsl_names = {
        "lesion_edge_ratio",
        "lesion_inner_ratio",
    }

    keep_idx = [
        i
        for i, name
        in enumerate(feature_names)
        if name not in lsl_names
    ]

    new_names = [
        feature_names[i]
        for i in keep_idx
    ]

    return (
        X_train[:, keep_idx],
        X_test[:, keep_idx],
        new_names,
    )


def run_one_experiment(
    experiment_name,
    X_train,
    y_train,
    X_test,
    y_test,
    n_features,
):
    """
    Fit StandardScaler on training data only, then fit/evaluate the fixed SVM.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = build_svm()

    model.fit(
        X_train_scaled,
        y_train,
    )

    train_pred = model.predict(
        X_train_scaled
    )

    test_pred = model.predict(
        X_test_scaled
    )

    train_metrics = evaluate_predictions(
        y_train,
        train_pred,
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_pred,
    )

    row = {
        "experiment": experiment_name,
        "n_features": int(n_features),

        "train_accuracy":
            train_metrics["accuracy"],

        "train_precision_macro":
            train_metrics["precision_macro"],

        "train_recall_macro":
            train_metrics["recall_macro"],

        "train_f1_macro":
            train_metrics["f1_macro"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_precision_macro":
            test_metrics["precision_macro"],

        "test_recall_macro":
            test_metrics["recall_macro"],

        "test_f1_macro":
            test_metrics["f1_macro"],
    }

    return row


def run_ablation(
    data_root,
    class_map_path,
    mask_root,
    cache_root,
    output_dir,
    include_wo_lsl=False,
):
    """Run all top-level feature-group ablations."""
    class_to_id, _, _ = load_class_mapping(
        class_map_path
    )

    rows = []

    full_cache = None

    for (
        experiment_name,
        feature_kwargs,
    ) in ABLATION_CONFIGS.items():

        (
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
        ) = load_or_extract_ablation_features(
            experiment_name=experiment_name,
            feature_kwargs=feature_kwargs,
            data_root=data_root,
            class_to_id=class_to_id,
            mask_root=mask_root,
            cache_root=cache_root,
        )

        if experiment_name == "Full":
            full_cache = (
                X_train,
                y_train,
                X_test,
                y_test,
                feature_names,
            )

        row = run_one_experiment(
            experiment_name=experiment_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_features=len(feature_names),
        )

        rows.append(row)

        print(
            f"{experiment_name:12s} | "
            f"features={len(feature_names):3d} | "
            f"test Acc="
            f"{row['test_accuracy'] * 100:.2f}% | "
            f"Macro-F1="
            f"{row['test_f1_macro'] * 100:.2f}%"
        )

    # Optional LSL-only ablation.
    if include_wo_lsl:
        if full_cache is None:
            raise RuntimeError(
                "Full feature cache is unavailable."
            )

        (
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
        ) = full_cache

        (
            X_train_wo_lsl,
            X_test_wo_lsl,
            names_wo_lsl,
        ) = remove_lsl_columns(
            X_train,
            X_test,
            feature_names,
        )

        row = run_one_experiment(
            experiment_name="w/o LSL",
            X_train=X_train_wo_lsl,
            y_train=y_train,
            X_test=X_test_wo_lsl,
            y_test=y_test,
            n_features=len(names_wo_lsl),
        )

        rows.append(row)

        print(
            f"{'w/o LSL':12s} | "
            f"features={len(names_wo_lsl):3d} | "
            f"test Acc="
            f"{row['test_accuracy'] * 100:.2f}% | "
            f"Macro-F1="
            f"{row['test_f1_macro'] * 100:.2f}%"
        )

    results_df = pd.DataFrame(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_dir / "ablation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "svm": SVM_PARAMS,
        "random_state": RANDOM_STATE,
        "feature_group_weighting_used": False,
        "experiments": ABLATION_CONFIGS,
        "optional_wo_lsl": bool(
            include_wo_lsl
        ),
    }

    with (
        output_dir / "ablation_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root containing train/ and test/.",
    )

    parser.add_argument(
        "--class-map",
        required=True,
        help="JSON containing {'class_to_id': {...}}.",
    )

    parser.add_argument(
        "--mask-root",
        default="masks",
    )

    parser.add_argument(
        "--cache-root",
        default="outputs/ablation_features",
    )

    parser.add_argument(
        "--output-dir",
        default="results/ablation",
    )

    parser.add_argument(
        "--include-wo-lsl",
        action="store_true",
        help=(
            "Also evaluate Full minus only "
            "lesion_edge_ratio and lesion_inner_ratio."
        ),
    )

    args = parser.parse_args()

    run_ablation(
        data_root=args.data_root,
        class_map_path=args.class_map,
        mask_root=args.mask_root,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        include_wo_lsl=args.include_wo_lsl,
    )


if __name__ == "__main__":
    main()
