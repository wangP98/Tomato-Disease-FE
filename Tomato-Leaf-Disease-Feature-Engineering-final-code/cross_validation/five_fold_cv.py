# -*- coding: utf-8 -*-
"""
Leakage-free stratified five-fold cross-validation for Full and Pruned SVMs.

Protocol
--------
1. Five-fold CV is performed ONLY on the original training set.
2. The independent test set is never loaded or used.
3. StandardScaler is fitted ONLY on the training subset of each fold.
4. Permutation importance is computed ONLY on the training subset of each fold.
5. Pearson-correlation pruning is computed ONLY on the training subset of each fold.
6. The held-out validation subset never participates in preprocessing,
   feature selection, or model fitting.
7. SVM hyperparameters are fixed and are not re-optimized within folds.
8. No feature-group weighting is used.
9. No feature-space augmentation is used in this validation analysis.

Fixed SVM configuration
-----------------------
kernel       = rbf
C            = 5.0
gamma        = scale
class_weight = balanced

Feature pruning
---------------
Permutation-importance threshold = 1e-4
Pearson |r| threshold            = 0.99
Permutation repeats              = 10
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
N_SPLITS = 5

SVM_C = 5.0
SVM_GAMMA = "scale"

IMP_THR = 1e-4
CORR_THR = 0.99
PERM_N_REPEATS = 10


def get_feature_group_name(name):
    """Map a feature name to one of the four manuscript feature groups."""
    if name.startswith("HSV_") or name.startswith("Lab_"):
        return "color"
    if name.startswith("LBP_feat_"):
        return "lbp"
    if name.startswith("lesion_"):
        return "lesion"
    return "shape"


def build_svm():
    """Build the fixed RBF-SVM used throughout the five folds."""
    return SVC(
        kernel="rbf",
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight="balanced",
        probability=False,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(y_true, y_pred):
    """Overall Accuracy and macro-averaged Precision/Recall/F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def calculate_classwise_metrics(
    y_true,
    y_pred,
    fold,
    model_name,
    id_to_class=None,
):
    """Return class-wise Precision, Recall, F1, and support for one fold."""
    labels = np.asarray(sorted(np.unique(y_true)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    rows = []
    for label, p, r, f, n in zip(
        labels, precision, recall, f1, support
    ):
        class_name = (
            id_to_class.get(int(label), str(label))
            if id_to_class is not None
            else str(label)
        )

        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "class_id": int(label),
                "class_name": class_name,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
        )

    return rows


def calculate_fold_class_distribution(
    y_train,
    y_val,
    fold,
    id_to_class=None,
):
    """Record train/validation class counts and percentages for one fold."""
    rows = []

    labels = sorted(
        np.unique(
            np.concatenate([y_train, y_val])
        )
    )

    for label in labels:
        train_count = int(np.sum(y_train == label))
        val_count = int(np.sum(y_val == label))

        class_name = (
            id_to_class.get(int(label), str(label))
            if id_to_class is not None
            else str(label)
        )

        rows.append(
            {
                "fold": fold,
                "class_id": int(label),
                "class_name": class_name,
                "train_count": train_count,
                "train_percent": train_count / len(y_train) * 100.0,
                "validation_count": val_count,
                "validation_percent": val_count / len(y_val) * 100.0,
            }
        )

    return rows


def select_features_foldwise(
    model,
    X_train,
    y_train,
    feature_names,
    fold,
    fold_dir=None,
    imp_thr=IMP_THR,
    corr_thr=CORR_THR,
):
    """
    Perform both pruning stages using ONLY the current fold-training data.

    Stage 1
        Permutation importance; retain mean PI > 1e-4.

    Stage 2
        Process Stage-1 features in descending PI order. A candidate is removed
        when |r| > 0.99 with any already-retained feature.
    """
    perm = permutation_importance(
        model,
        X_train,
        y_train,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="f1_macro",
    )

    fi_df = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "perm_importance_mean": perm.importances_mean,
                "perm_importance_std": perm.importances_std,
            }
        )
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    keep_feats = fi_df.loc[
        fi_df["perm_importance_mean"] > imp_thr,
        "feature",
    ].tolist()

    if len(keep_feats) == 0:
        keep_feats = fi_df["feature"].tolist()

    n_after_pi = len(keep_feats)

    df_train = pd.DataFrame(
        X_train,
        columns=feature_names,
    )
    sub_corr = df_train[keep_feats].corr(method="pearson")

    sorted_imp = (
        fi_df.set_index("feature")
        .loc[keep_feats, "perm_importance_mean"]
        .sort_values(ascending=False)
    )

    selected = []

    for feature in sorted_imp.index:
        if not selected:
            selected.append(feature)
            continue

        drop_flag = False

        for retained in selected:
            corr_value = sub_corr.loc[feature, retained]

            if (
                np.isfinite(corr_value)
                and abs(corr_value) > corr_thr
            ):
                drop_flag = True
                break

        if not drop_flag:
            selected.append(feature)

    group_counts = {
        "color": 0,
        "lbp": 0,
        "shape": 0,
        "lesion": 0,
    }
    for feature in selected:
        group_counts[get_feature_group_name(feature)] += 1

    if fold_dir is not None:
        fold_dir = Path(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        fi_df.to_csv(
            fold_dir / "permutation_importance_train_only.csv",
            index=False,
            encoding="utf-8-sig",
        )

        sub_corr.to_csv(
            fold_dir / "pearson_correlation_train_only.csv",
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            {
                "feature": selected,
                "group": [
                    get_feature_group_name(f)
                    for f in selected
                ],
            }
        ).to_csv(
            fold_dir / "selected_features.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return selected, fi_df, n_after_pi, group_counts


def summarize_metrics(df):
    """Mean ± sample SD across five folds."""
    rows = []

    for model_name, group in df.groupby("model", sort=False):
        row = {"model": model_name}

        for metric in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]:
            mean_value = float(group[metric].mean())
            std_value = float(group[metric].std(ddof=1))

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_paper"] = (
                f"{mean_value * 100:.2f} ± "
                f"{std_value * 100:.2f}"
            )

        rows.append(row)

    return pd.DataFrame(rows)


def run_five_fold_cv(
    X,
    y,
    feature_names,
    output_dir=None,
    id_to_class=None,
):
    """
    Run the complete reviewer-oriented five-fold validation analysis.

    Parameters
    ----------
    X : ndarray
        Original 296-D training feature matrix.
    y : ndarray
        Original training labels.
    feature_names : sequence of str
        Names corresponding to columns of X.
    output_dir : str or Path, optional
        If provided, detailed fold-wise CSV outputs and config.json are saved.
    id_to_class : dict, optional
        Mapping from numeric labels to class names.

    Returns
    -------
    results : dict
        DataFrames for fold metrics, summaries, class-wise metrics, class
        distributions, pruning counts, selection frequencies, and OOF predictions.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    feature_names = list(feature_names)

    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    if len(X) != len(y):
        raise ValueError("X/y sample-count mismatch.")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature dimension mismatch: {X.shape[1]} vs {len(feature_names)}."
        )

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []
    classwise_results = []
    oof_results = []
    fold_distribution = []
    pruning_results = []
    selected_features_all_folds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        fold_dir = (
            output_dir / f"fold_{fold}"
            if output_dir is not None
            else None
        )

        X_train_raw = X[train_idx]
        y_train = y[train_idx]
        X_val_raw = X[val_idx]
        y_val = y[val_idx]

        fold_distribution.extend(
            calculate_fold_class_distribution(
                y_train,
                y_val,
                fold,
                id_to_class=id_to_class,
            )
        )

        # Fit StandardScaler only on current fold-training subset.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        # Full SVM.
        full_model = build_svm()
        full_model.fit(X_train_scaled, y_train)
        y_pred_full = full_model.predict(X_val_scaled)

        full_metrics = calculate_metrics(
            y_val,
            y_pred_full,
        )

        fold_results.append(
            {
                "model": "Full SVM",
                "fold": fold,
                "n_features": len(feature_names),
                **full_metrics,
            }
        )

        classwise_results.extend(
            calculate_classwise_metrics(
                y_val,
                y_pred_full,
                fold,
                "Full SVM",
                id_to_class=id_to_class,
            )
        )

        # Fold-wise feature selection using fold-training only.
        (
            pruned_features,
            fi_df,
            n_after_pi,
            group_counts,
        ) = select_features_foldwise(
            model=full_model,
            X_train=X_train_scaled,
            y_train=y_train,
            feature_names=feature_names,
            fold=fold,
            fold_dir=fold_dir,
            imp_thr=IMP_THR,
            corr_thr=CORR_THR,
        )

        feature_index_map = {
            name: i
            for i, name in enumerate(feature_names)
        }
        pruned_idx = np.asarray(
            [
                feature_index_map[f]
                for f in pruned_features
            ],
            dtype=int,
        )

        X_train_pruned = X_train_scaled[:, pruned_idx]
        X_val_pruned = X_val_scaled[:, pruned_idx]

        # Pruned SVM: same fixed SVM configuration, no augmentation.
        pruned_model = build_svm()
        pruned_model.fit(
            X_train_pruned,
            y_train,
        )
        y_pred_pruned = pruned_model.predict(
            X_val_pruned
        )

        pruned_metrics = calculate_metrics(
            y_val,
            y_pred_pruned,
        )

        fold_results.append(
            {
                "model": "Pruned SVM",
                "fold": fold,
                "n_features": len(pruned_features),
                **pruned_metrics,
            }
        )

        classwise_results.extend(
            calculate_classwise_metrics(
                y_val,
                y_pred_pruned,
                fold,
                "Pruned SVM",
                id_to_class=id_to_class,
            )
        )

        pruning_results.append(
            {
                "fold": fold,
                "features_before": len(feature_names),
                "features_after_PI": n_after_pi,
                "features_after_Pearson": len(pruned_features),
                "reduction_percent": (
                    1.0
                    - len(pruned_features) / len(feature_names)
                ) * 100.0,
                "color_features": group_counts["color"],
                "lbp_features": group_counts["lbp"],
                "shape_features": group_counts["shape"],
                "lesion_features": group_counts["lesion"],
            }
        )

        for feature in pruned_features:
            selected_features_all_folds.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "group": get_feature_group_name(feature),
                }
            )

        for (
            original_idx,
            true_label,
            pred_full,
            pred_pruned,
        ) in zip(
            val_idx,
            y_val,
            y_pred_full,
            y_pred_pruned,
        ):
            true_class = (
                id_to_class.get(int(true_label), str(true_label))
                if id_to_class is not None
                else str(true_label)
            )
            full_class = (
                id_to_class.get(int(pred_full), str(pred_full))
                if id_to_class is not None
                else str(pred_full)
            )
            pruned_class = (
                id_to_class.get(int(pred_pruned), str(pred_pruned))
                if id_to_class is not None
                else str(pred_pruned)
            )

            oof_results.append(
                {
                    "sample_index": int(original_idx),
                    "fold": fold,
                    "y_true": int(true_label),
                    "true_class": true_class,
                    "full_pred": int(pred_full),
                    "full_pred_class": full_class,
                    "full_correct": int(true_label == pred_full),
                    "pruned_pred": int(pred_pruned),
                    "pruned_pred_class": pruned_class,
                    "pruned_correct": int(true_label == pred_pruned),
                }
            )

    fold_df = pd.DataFrame(fold_results)
    classwise_df = pd.DataFrame(classwise_results)
    oof_df = pd.DataFrame(oof_results).sort_values("sample_index")
    distribution_df = pd.DataFrame(fold_distribution)
    pruning_df = pd.DataFrame(pruning_results)
    selected_all_df = pd.DataFrame(selected_features_all_folds)

    summary_df = summarize_metrics(fold_df)

    pruned_feature_counts = pruning_df[
        "features_after_Pearson"
    ]

    feature_count_summary = pd.DataFrame(
        [
            {
                "mean_selected_features": float(
                    pruned_feature_counts.mean()
                ),
                "std_selected_features": float(
                    pruned_feature_counts.std(ddof=1)
                ),
                "min_selected_features": int(
                    pruned_feature_counts.min()
                ),
                "max_selected_features": int(
                    pruned_feature_counts.max()
                ),
                "mean_reduction_percent": float(
                    pruning_df["reduction_percent"].mean()
                ),
            }
        ]
    )

    if not selected_all_df.empty:
        feature_frequency_df = (
            selected_all_df
            .groupby(["feature", "group"])
            .size()
            .reset_index(name="selected_folds")
            .sort_values(
                ["selected_folds", "feature"],
                ascending=[False, True],
            )
        )
        feature_frequency_df[
            "selection_frequency_percent"
        ] = (
            feature_frequency_df["selected_folds"]
            / N_SPLITS
            * 100.0
        )
    else:
        feature_frequency_df = pd.DataFrame()

    class_summary_rows = []

    for (
        model_name,
        class_id,
        class_name,
    ), group_df in classwise_df.groupby(
        ["model", "class_id", "class_name"]
    ):
        row = {
            "model": model_name,
            "class_id": class_id,
            "class_name": class_name,
        }

        for metric in ["precision", "recall", "f1"]:
            mean_value = float(group_df[metric].mean())
            std_value = float(group_df[metric].std(ddof=1))

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_paper"] = (
                f"{mean_value * 100:.2f} ± "
                f"{std_value * 100:.2f}"
            )

        class_summary_rows.append(row)

    class_summary_df = pd.DataFrame(
        class_summary_rows
    )

    if output_dir is not None:
        fold_df.to_csv(
            output_dir / "cv_fold_metrics_full_pruned.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_df.to_csv(
            output_dir / "cv_summary_full_pruned.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pruning_df.to_csv(
            output_dir / "foldwise_pruning_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        feature_count_summary.to_csv(
            output_dir / "pruned_feature_count_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        selected_all_df.to_csv(
            output_dir / "selected_features_all_folds.csv",
            index=False,
            encoding="utf-8-sig",
        )

        feature_frequency_df.to_csv(
            output_dir / "feature_selection_frequency.csv",
            index=False,
            encoding="utf-8-sig",
        )

        classwise_df.to_csv(
            output_dir / "cv_classwise_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        class_summary_df.to_csv(
            output_dir / "cv_classwise_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        distribution_df.to_csv(
            output_dir / "fold_class_distribution.csv",
            index=False,
            encoding="utf-8-sig",
        )

        oof_df.to_csv(
            output_dir / "cv_oof_predictions_full_pruned.csv",
            index=False,
            encoding="utf-8-sig",
        )

        config = {
            "method": (
                "Stratified 5-fold CV: "
                "Full SVM + fold-wise Pruned SVM"
            ),
            "scope": "training set only",
            "test_set_used": False,
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "n_samples": int(len(y)),
            "original_features": int(len(feature_names)),
            "svm": {
                "kernel": "rbf",
                "C": SVM_C,
                "gamma": SVM_GAMMA,
                "class_weight": "balanced",
            },
            "feature_group_weighting_used": False,
            "feature_augmentation_used": False,
            "feature_selection": {
                "performed_inside_each_fold": True,
                "permutation_importance_threshold": IMP_THR,
                "permutation_repeats": PERM_N_REPEATS,
                "permutation_scoring": "f1_macro",
                "pearson_abs_correlation_threshold": CORR_THR,
                "validation_used_for_feature_selection": False,
            },
            "preprocessing": {
                "scaler": "StandardScaler",
                "scaler_fit_scope": (
                    "fold training subset only"
                ),
            },
            "hyperparameter_tuning_inside_fold": False,
            "note": (
                "SVM hyperparameters were fixed. "
                "Standardization and the complete two-stage "
                "feature-selection procedure were repeated "
                "independently using only the training subset "
                "of each fold."
            ),
        }

        with (
            output_dir / "config.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(
                config,
                f,
                ensure_ascii=False,
                indent=4,
            )

    return {
        "fold_metrics": fold_df,
        "summary": summary_df,
        "pruning": pruning_df,
        "feature_count_summary": feature_count_summary,
        "selected_features_all_folds": selected_all_df,
        "feature_frequency": feature_frequency_df,
        "classwise_metrics": classwise_df,
        "classwise_summary": class_summary_df,
        "class_distribution": distribution_df,
        "oof_predictions": oof_df,
    }
