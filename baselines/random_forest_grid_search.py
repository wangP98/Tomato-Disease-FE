# -*- coding: utf-8 -*-
"""Random-Forest baseline with TRAIN-only stratified five-fold parameter search."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification.evaluate_svm import evaluate_model
from utils.class_mapping import load_class_mapping
from utils.dataset_loader import load_split_features


RANDOM_STATE = 42
N_SPLITS = 5

FEATURE_CONFIG = dict(
    use_color=True,
    use_lbp=True,
    use_shape=True,
    use_lesion=True,
)

PARAM_GRID = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
    "rf__max_features": ["sqrt", "log2"],
}


def run_rf_search(
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

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # RF does not require feature scaling, but StandardScaler is retained
    # here to match the preprocessing convention of the original comparison.
    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

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

    search.fit(
        X_train,
        y_train,
    )

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

    pd.DataFrame(
        search.cv_results_
    ).sort_values(
        "rank_test_score"
    ).to_csv(
        output_dir
        / "rf_grid_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        output_dir
        / "rf_best_params.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "selection_metric": "f1_macro",
                "n_splits": N_SPLITS,
                "shuffle": True,
                "random_state": RANDOM_STATE,
                "fixed_class_weight": "balanced",
                "best_cv_macro_f1": float(
                    search.best_score_
                ),
                "best_params": search.best_params_,
                "param_grid": PARAM_GRID,
                "test_set_used_for_search": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_model = search.best_estimator_

    evaluate_model(
        best_model,
        X_test,
        y_test,
        class_to_id,
        "Test_RF",
        output_dir,
    )

    joblib.dump(
        best_model,
        output_dir
        / "rf_best_pipeline.joblib",
    )

    print(
        "Best CV Macro-F1:",
        search.best_score_,
    )
    print(
        "Best parameters:",
        search.best_params_,
    )
    print(
        "Saved to:",
        output_dir,
    )


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
        default="outputs/baselines/rf_grid_search",
    )
    args = parser.parse_args()

    run_rf_search(
        data_root=args.data_root,
        class_map_path=args.class_map,
        mask_root=args.mask_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
