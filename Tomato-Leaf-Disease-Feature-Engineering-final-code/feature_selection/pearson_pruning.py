# -*- coding: utf-8 -*-
"""Stage 2 of the two-stage pruning procedure: Pearson redundancy pruning."""

from pathlib import Path

import pandas as pd

DEFAULT_CORRELATION_THRESHOLD = 0.99


def pearson_redundancy_pruning(
    X_train,
    feature_names,
    importance_df,
    stage1_features,
    threshold=DEFAULT_CORRELATION_THRESHOLD,
):
    """
    Greedily retain high-importance features while removing highly correlated
    lower-ranked features.

    Features are processed in descending permutation-importance order.
    A candidate is removed when |r| > threshold with any already retained
    feature.
    """
    feature_names = list(feature_names)
    stage1_features = list(stage1_features)

    train_df = pd.DataFrame(
        X_train,
        columns=feature_names,
    )
    corr = train_df[stage1_features].corr(method="pearson")

    importance_rank = (
        importance_df.set_index("feature")
        .loc[stage1_features, "perm_importance_mean"]
        .sort_values(ascending=False)
    )

    selected = []

    for feature in importance_rank.index:
        if not selected:
            selected.append(feature)
            continue

        should_drop = any(
            abs(corr.loc[feature, retained_feature]) > threshold
            for retained_feature in selected
        )

        if not should_drop:
            selected.append(feature)

    return selected, corr


def save_pearson_pruning(
    selected_features,
    correlation_matrix,
    output_dir,
):
    """Save the Stage-2 correlation matrix and final selected-feature list."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correlation_matrix.to_csv(
        output_dir / "pearson_correlation_stage1.csv",
        encoding="utf-8-sig",
    )

    pd.Series(
        selected_features,
        name="feature",
    ).to_csv(
        output_dir / "selected_features_final.csv",
        index=False,
        encoding="utf-8-sig",
    )
