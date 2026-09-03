# -*- coding: utf-8 -*-
"""
SVM baseline with TRAIN-only stratified five-fold GridSearchCV.

Purpose
-------
Search the hyperparameters of the RBF-SVM using the same 296-D handcrafted
feature representation as the Full SVM.

Protocol
--------
1. Extract Full features:
       Color + LBP + Shape + Lesion = 296 dimensions
2. Use only the TRAIN partition for hyperparameter search.
3. StandardScaler is fitted inside each CV fold through sklearn Pipeline.
4. Use StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
5. Select the best hyperparameters according to Macro-F1.
6. Refit the best pipeline on the complete TRAIN partition.
7. Evaluate the independent TEST partition only once after model selection.

No feature-group weighting is used.
No permutation-importance or Pearson pruning is used in this baseline search.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classification.evaluate_svm import evaluate_model
from utils.class_mapping import load_class_mapping
from utils.dataset_loader import load_split_features


# ============================================================
# Reproducibility
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# Full handcrafted feature configuration
# ============================================================

FEATURE_CONFIG = dict(
    use_color=True,
    use_lbp=True,
    use_shape=True,
    use_lesion=True,
)


# ============================================================
# SVM search space
# ============================================================
#
# The main manuscript SVM uses an RBF kernel, so the grid search
# keeps the kernel fixed as RBF and searches C and gamma.
#
# ============================================================

PARAM_GRID = {
    "svm__C": [
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        50.0,
    ],

    "svm__gamma": [
        "scale",
        "auto",
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ],
}


# ============================================================
# Main search function
# ============================================================

def run_svm_search(
    data_root,
    class_map_path,
    mask_root,
    output_root,
):
    data_root = Path(
        data_root
    )

    # --------------------------------------------------------
    # 1. Class mapping
    # --------------------------------------------------------

    (
        class_to_id,
        _,
        _,
    ) = load_class_mapping(
        class_map_path
    )

    # --------------------------------------------------------
    # 2. Extract Full 296-D TRAIN features
    # --------------------------------------------------------

    print(
        "\n[INFO] Extracting TRAIN features..."
    )

    X_train, y_train = load_split_features(
        data_root / "train",
        class_to_id=class_to_id,
        mask_root=mask_root,
        common_root=data_root,
        **FEATURE_CONFIG,
    )

    # --------------------------------------------------------
    # 3. Extract independent TEST features
    #
    # TEST is loaded here for final evaluation only.
    # It is never passed to GridSearchCV.
    # --------------------------------------------------------

    print(
        "\n[INFO] Extracting TEST features..."
    )

    X_test, y_test = load_split_features(
        data_root / "test",
        class_to_id=class_to_id,
        mask_root=mask_root,
        common_root=data_root,
        **FEATURE_CONFIG,
    )

    print(
        f"\nTRAIN feature matrix: "
        f"{X_train.shape}"
    )

    print(
        f"TEST feature matrix : "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # 4. Stratified five-fold CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # 5. Leakage-free preprocessing + SVM
    #
    # StandardScaler is inside Pipeline, so every CV fold fits
    # its scaler only on the fold-training subset.
    # --------------------------------------------------------

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # 6. GridSearchCV
    # --------------------------------------------------------

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2,
        return_train_score=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SVM GRID SEARCH"
    )

    print(
        "=" * 80
    )

    print(
        f"CV folds               : "
        f"{N_SPLITS}"
    )

    print(
        f"Selection metric       : "
        f"Macro-F1"
    )

    print(
        f"Parameter combinations : "
        f"{len(PARAM_GRID['svm__C']) * len(PARAM_GRID['svm__gamma'])}"
    )

    print(
        "TEST used for search   : "
        "False"
    )

    print(
        "=" * 80
    )

    search.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 7. Create output directory
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 8. Save complete grid-search table
    # --------------------------------------------------------

    cv_results_df = pd.DataFrame(
        search.cv_results_
    )

    cv_results_df = (
        cv_results_df
        .sort_values(
            [
                "rank_test_score",
                "mean_test_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    cv_results_path = (
        output_dir
        / "svm_grid_search_results.csv"
    )

    cv_results_df.to_csv(
        cv_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 9. Save compact top-ranking table
    # --------------------------------------------------------

    useful_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_svm__C",
        "param_svm__gamma",
    ]

    top_results_df = (
        cv_results_df[
            useful_columns
        ]
        .head(20)
        .copy()
    )

    top_results_df.to_csv(
        output_dir
        / "svm_grid_search_top20.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 10. Save best configuration
    # --------------------------------------------------------

    best_config = {
        "search_method":
            "GridSearchCV",

        "classifier":
            "SVC",

        "kernel":
            "rbf",

        "class_weight":
            "balanced",

        "selection_metric":
            "f1_macro",

        "n_splits":
            N_SPLITS,

        "shuffle":
            True,

        "random_state":
            RANDOM_STATE,

        "feature_configuration":
            FEATURE_CONFIG,

        "feature_pruning_used":
            False,

        "feature_group_weighting_used":
            False,

        "best_cv_macro_f1":
            float(
                search.best_score_
            ),

        "best_params":
            search.best_params_,

        "param_grid":
            PARAM_GRID,

        "test_set_used_for_search":
            False,
    }

    with (
        output_dir
        / "svm_best_params.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 11. Best pipeline
    #
    # GridSearchCV(refit=True) has already refitted this model
    # on the complete TRAIN partition.
    # --------------------------------------------------------

    best_model = (
        search.best_estimator_
    )

    # --------------------------------------------------------
    # 12. Independent TEST evaluation
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST SVM"
    )

    print(
        "=" * 80
    )

    print(
        f"Best CV Macro-F1 : "
        f"{search.best_score_:.6f}"
    )

    print(
        "Best parameters:"
    )

    for key, value in (
        search.best_params_.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        "\nEvaluating best model "
        "on independent TEST set..."
    )

    (
        y_test_pred,
        test_metrics,
        _
    ) = evaluate_model(
        best_model,
        X_test,
        y_test,
        class_to_id,
        "Test_SVM_GridSearch",
        output_dir,
    )

    # --------------------------------------------------------
    # 13. Save final TRAIN-search / TEST summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "Best_C":
                    search.best_params_[
                        "svm__C"
                    ],

                "Best_gamma":
                    search.best_params_[
                        "svm__gamma"
                    ],

                "CV_Macro_F1":
                    search.best_score_,

                "Test_Accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "Test_Macro_Precision":
                    test_metrics[
                        "precision_macro"
                    ],

                "Test_Macro_Recall":
                    test_metrics[
                        "recall_macro"
                    ],

                "Test_Macro_F1":
                    test_metrics[
                        "f1_macro"
                    ],
            }
        ]
    )

    summary_df.to_csv(
        output_dir
        / "svm_grid_search_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 14. Save fitted best Pipeline
    #
    # Contains:
    # - fitted StandardScaler
    # - fitted best RBF-SVM
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        output_dir
        / "svm_best_pipeline.joblib",
    )

    print(
        "\nTEST metrics:"
    )

    print(
        f"  Accuracy        : "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro Precision : "
        f"{test_metrics['precision_macro'] * 100:.2f}%"
    )

    print(
        f"  Macro Recall    : "
        f"{test_metrics['recall_macro'] * 100:.2f}%"
    )

    print(
        f"  Macro F1        : "
        f"{test_metrics['f1_macro'] * 100:.2f}%"
    )

    print(
        "\nSaved to:"
    )

    print(
        output_dir
    )

    return (
        search,
        test_metrics,
        output_dir,
    )


# ============================================================
# Command line
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RBF-SVM parameter search using "
            "TRAIN-only stratified five-fold CV."
        )
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Dataset root containing "
            "train/ and test/."
        ),
    )

    parser.add_argument(
        "--class-map",
        default=(
            "datasets/"
            "class_to_id.json"
        ),
    )

    parser.add_argument(
        "--mask-root",
        default="masks",
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/"
            "baselines/"
            "svm_grid_search"
        ),
    )

    args = parser.parse_args()

    run_svm_search(
        data_root=args.data_root,
        class_map_path=args.class_map,
        mask_root=args.mask_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
