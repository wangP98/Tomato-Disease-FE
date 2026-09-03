# -*- coding: utf-8 -*-
"""Stage 1 of the two-stage pruning procedure: permutation importance."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

DEFAULT_IMPORTANCE_THRESHOLD = 1e-4
DEFAULT_N_REPEATS = 10
DEFAULT_RANDOM_STATE = 42


def compute_permutation_importance(
    model,
    X_train,
    y_train,
    feature_names,
    n_repeats=DEFAULT_N_REPEATS,
    random_state=DEFAULT_RANDOM_STATE,
    scoring="f1_macro",
):
    """
    Compute permutation importance using training data only.
    """
    result = permutation_importance(
        model,
        X_train,
        y_train,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring,
    )

    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "perm_importance_mean": result.importances_mean,
                "perm_importance_std": result.importances_std,
            }
        )
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def retain_by_importance(
    importance_df,
    threshold=DEFAULT_IMPORTANCE_THRESHOLD,
):
    """
    Retain features with mean permutation importance > threshold.

    The manuscript threshold is 1e-4.
    """
    retained = importance_df.loc[
        importance_df["perm_importance_mean"] > threshold,
        "feature",
    ].tolist()

    if not retained:
        retained = importance_df["feature"].tolist()

    return retained


def save_permutation_importance(
    importance_df,
    output_dir,
    top_n=30,
):
    """Save the numerical importance table and an optional top-N plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "permutation_importance.csv"
    importance_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    top_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top_df["feature"][::-1],
        top_df["perm_importance_mean"][::-1],
    )
    ax.set_title(
        f"Top {top_n} Permutation Importances (Macro-F1)"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "permutation_importance_top30.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return csv_path
