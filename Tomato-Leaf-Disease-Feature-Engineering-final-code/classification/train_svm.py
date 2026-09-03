# -*- coding: utf-8 -*-
"""
Train Full and Pruned RBF-SVM models without feature-group weighting.

The supplied historical script had Color/LBP/Shape/Lesion weights all set to
1.0, so the weighting operation was numerically an identity transformation.
This cleaned version removes that redundant mechanism entirely.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction.feature_names import build_feature_names
from feature_selection import (
    compute_permutation_importance,
    retain_by_importance,
    save_permutation_importance,
    pearson_redundancy_pruning,
    save_pearson_pruning,
)
from utils.class_mapping import load_class_mapping
from utils.dataset_loader import load_split_features
from utils.feature_augmentation import (
    augment_features,
    get_color_feature_indices,
)
from classification.evaluate_svm import evaluate_model


RANDOM_STATE = 42

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 5.0,
    "gamma": "scale",
    "class_weight": "balanced",
}

IMPORTANCE_THRESHOLD = 1e-4
CORRELATION_THRESHOLD = 0.99
PERMUTATION_REPEATS = 10

# Preserved from the supplied pruned-model training implementation.
USE_PRUNED_AUGMENTATION = True
AUG_N = 1
AUG_NOISE_STD = 0.01
AUG_BRIGHT_RANGE = (0.9, 1.1)


def build_svm(probability=True):
    """Construct the fixed RBF-SVM used in the study."""
    return SVC(
        kernel=SVM_PARAMS["kernel"],
        C=SVM_PARAMS["C"],
        gamma=SVM_PARAMS["gamma"],
        class_weight=SVM_PARAMS["class_weight"],
        probability=probability,
        random_state=RANDOM_STATE,
    )


def load_or_extract_features(
    split_name,
    split_dir,
    class_to_id,
    feature_names,
    cache_dir,
    mask_root,
    common_root,
):
    """Load cached feature matrices or extract them when missing."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    x_path = cache_dir / f"X_{split_name}_raw.npy"
    y_path = cache_dir / f"y_{split_name}.npy"

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        X, y = load_split_features(
            split_dir,
            class_to_id=class_to_id,
            mask_root=mask_root,
            common_root=common_root,
            use_color=True,
            use_lbp=True,
            use_shape=True,
            use_lesion=True,
        )
        np.save(x_path, X)
        np.save(y_path, y)

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"{split_name}: feature dimension {X.shape[1]} "
            f"does not match {len(feature_names)} feature names."
        )

    return np.asarray(X), np.asarray(y)


def save_pruned_feature_matrix(
    X,
    feature_names,
    output_path,
):
    """Save a pruned feature matrix as both NPY and CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, X)
    pd.DataFrame(
        X,
        columns=feature_names,
    ).to_csv(
        output_path.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )


def train_pipeline(
    data_root,
    class_map_path,
    mask_root,
    feature_cache_dir,
    output_root,
):
    """Train/evaluate Full SVM, perform two-stage pruning, then train Pruned SVM."""
    data_root = Path(data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    class_to_id, id_to_class, _ = load_class_mapping(
        class_map_path
    )

    feature_names = build_feature_names(
        use_color=True,
        use_lbp=True,
        use_shape=True,
        use_lesion=True,
    )

    X_train, y_train = load_or_extract_features(
        "train",
        train_dir,
        class_to_id,
        feature_names,
        feature_cache_dir,
        mask_root,
        data_root,
    )
    X_test, y_test = load_or_extract_features(
        "test",
        test_dir,
        class_to_id,
        feature_names,
        feature_cache_dir,
        mask_root,
        data_root,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Standardization: fit on training data only.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Full SVM.
    # ------------------------------------------------------------------
    full_model = build_svm(probability=True)
    full_model.fit(X_train_scaled, y_train)

    evaluate_model(
        full_model,
        X_train_scaled,
        y_train,
        class_to_id,
        "Train_full",
        run_dir,
    )
    evaluate_model(
        full_model,
        X_test_scaled,
        y_test,
        class_to_id,
        "Test_full",
        run_dir,
    )

    # ------------------------------------------------------------------
    # Stage 1: permutation importance on training data only.
    # ------------------------------------------------------------------
    importance_df = compute_permutation_importance(
        full_model,
        X_train_scaled,
        y_train,
        feature_names,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
    )
    save_permutation_importance(
        importance_df,
        run_dir / "feature_selection",
    )

    stage1_features = retain_by_importance(
        importance_df,
        threshold=IMPORTANCE_THRESHOLD,
    )
    pd.Series(
        stage1_features,
        name="feature",
    ).to_csv(
        run_dir / "feature_selection" / "stage1_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Stage 2: Pearson redundancy pruning on training data only.
    # ------------------------------------------------------------------
    selected_features, corr_matrix = pearson_redundancy_pruning(
        X_train_scaled,
        feature_names,
        importance_df,
        stage1_features,
        threshold=CORRELATION_THRESHOLD,
    )
    save_pearson_pruning(
        selected_features,
        corr_matrix,
        run_dir / "feature_selection",
    )

    feature_index = {
        name: i
        for i, name in enumerate(feature_names)
    }
    selected_idx = np.asarray(
        [feature_index[name] for name in selected_features],
        dtype=int,
    )

    X_train_pruned = X_train_scaled[:, selected_idx]
    X_test_pruned = X_test_scaled[:, selected_idx]

    save_pruned_feature_matrix(
        X_train_pruned,
        selected_features,
        Path(feature_cache_dir) / "X_train_pruned.npy",
    )
    save_pruned_feature_matrix(
        X_test_pruned,
        selected_features,
        Path(feature_cache_dir) / "X_test_pruned.npy",
    )

    # ------------------------------------------------------------------
    # Pruned SVM.
    # The feature-space augmentation behavior is preserved from the
    # supplied historical training implementation.
    # ------------------------------------------------------------------
    if USE_PRUNED_AUGMENTATION:
        pruned_color_idx = get_color_feature_indices(
            selected_features
        )
        X_train_pruned_fit, y_train_pruned_fit = augment_features(
            X_train_pruned,
            y_train,
            n_aug=AUG_N,
            noise_std=AUG_NOISE_STD,
            color_idx=pruned_color_idx,
            bright_range=AUG_BRIGHT_RANGE,
            random_state=RANDOM_STATE,
        )
    else:
        X_train_pruned_fit = X_train_pruned
        y_train_pruned_fit = y_train

    pruned_model = build_svm(probability=True)
    pruned_model.fit(
        X_train_pruned_fit,
        y_train_pruned_fit,
    )

    evaluate_model(
        pruned_model,
        X_train_pruned,
        y_train,
        class_to_id,
        "Train_pruned",
        run_dir,
    )
    evaluate_model(
        pruned_model,
        X_test_pruned,
        y_test,
        class_to_id,
        "Test_pruned",
        run_dir,
    )

    config = {
        "random_state": RANDOM_STATE,
        "svm": SVM_PARAMS,
        "feature_group_weighting_used": False,
        "feature_selection": {
            "permutation_repeats": PERMUTATION_REPEATS,
            "importance_threshold": IMPORTANCE_THRESHOLD,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "n_features_before": len(feature_names),
            "n_features_after_stage1": len(stage1_features),
            "n_features_final": len(selected_features),
        },
        "pruned_feature_augmentation": {
            "enabled": USE_PRUNED_AUGMENTATION,
            "n_aug": AUG_N,
            "noise_std": AUG_NOISE_STD,
            "color_brightness_range": list(AUG_BRIGHT_RANGE),
        },
    }

    with (run_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(
        {
            "scaler": scaler,
            "model_full": full_model,
            "model_pruned": pruned_model,
            "feature_names": feature_names,
            "pruned_feature_names": selected_features,
            "class_to_id": class_to_id,
            "id_to_class": id_to_class,
            "random_state": RANDOM_STATE,
            "svm_params": SVM_PARAMS,
            "feature_group_weighting_used": False,
        },
        run_dir / "svm_full_and_pruned.joblib",
    )

    print(f"Run saved to: {run_dir}")
    print(
        f"Feature pruning: {len(feature_names)} "
        f"-> {len(stage1_features)} "
        f"-> {len(selected_features)}"
    )

    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Full/Pruned SVM without feature-group weighting."
    )
    parser.add_argument(
        "--data-root",
        default="data01",
        help="Dataset root containing train/ and test/.",
    )
    parser.add_argument(
        "--class-map",
        default="datasets/class_to_id.json",
        help="JSON file containing class_to_id.",
    )
    parser.add_argument(
        "--mask-root",
        default="masks",
        help="Leaf-mask cache directory.",
    )
    parser.add_argument(
        "--feature-cache",
        default="outputs/features",
        help="Feature cache directory.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/training",
        help="Training output root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        data_root=args.data_root,
        class_map_path=args.class_map,
        mask_root=args.mask_root,
        feature_cache_dir=args.feature_cache,
        output_root=args.output_root,
    )
