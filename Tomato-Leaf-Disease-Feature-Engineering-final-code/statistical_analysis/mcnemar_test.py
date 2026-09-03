# -*- coding: utf-8 -*-
"""
Full SVM vs Pruned SVM paired McNemar analysis.

The manuscript reports:
- discordant prediction counts,
- the uncorrected McNemar chi-square statistic,
- the two-sided exact McNemar p-value,
- alpha = 0.05.

The continuity-corrected chi-square statistic is also available here as
supplemental numerical output, but it is not used as the primary inference.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score


def calculate_mcnemar(
    y_true,
    full_pred,
    pruned_pred,
):
    """Calculate paired prediction counts and McNemar statistics."""
    y_true = np.asarray(y_true).reshape(-1)
    full_pred = np.asarray(full_pred).reshape(-1)
    pruned_pred = np.asarray(pruned_pred).reshape(-1)

    if not (
        len(y_true)
        == len(full_pred)
        == len(pruned_pred)
    ):
        raise ValueError(
            "y_true, full_pred, and pruned_pred must have equal lengths."
        )

    full_correct = full_pred == y_true
    pruned_correct = pruned_pred == y_true

    both_correct = int(
        np.sum(full_correct & pruned_correct)
    )
    full_only_correct = int(
        np.sum(full_correct & (~pruned_correct))
    )
    pruned_only_correct = int(
        np.sum((~full_correct) & pruned_correct)
    )
    both_wrong = int(
        np.sum((~full_correct) & (~pruned_correct))
    )

    discordant = (
        full_only_correct
        + pruned_only_correct
    )

    if discordant == 0:
        exact_p = 1.0
        chi2_uncorrected = 0.0
        chi2_uncorrected_p = 1.0
        chi2_corrected = 0.0
        chi2_corrected_p = 1.0
    else:
        exact_p = float(
            binomtest(
                k=full_only_correct,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

        chi2_uncorrected = (
            (full_only_correct - pruned_only_correct) ** 2
            / discordant
        )
        chi2_uncorrected_p = float(
            chi2.sf(
                chi2_uncorrected,
                df=1,
            )
        )

        chi2_corrected = (
            (
                abs(
                    full_only_correct
                    - pruned_only_correct
                )
                - 1
            ) ** 2
            / discordant
        )
        chi2_corrected_p = float(
            chi2.sf(
                chi2_corrected,
                df=1,
            )
        )

    return {
        "N": int(len(y_true)),
        "Both correct": both_correct,
        "Full correct / Pruned wrong": full_only_correct,
        "Full wrong / Pruned correct": pruned_only_correct,
        "Both wrong": both_wrong,
        "Discordant pairs": int(discordant),
        "Uncorrected McNemar chi-square": float(
            chi2_uncorrected
        ),
        "Uncorrected chi-square p-value": float(
            chi2_uncorrected_p
        ),
        "Exact McNemar p-value": exact_p,
        "Corrected McNemar chi-square": float(
            chi2_corrected
        ),
        "Corrected chi-square p-value": float(
            chi2_corrected_p
        ),
    }


def predict_from_saved_models(
    model_file,
    X_test_file,
    y_test_file,
):
    """
    Reconstruct Full/Pruned predictions from a saved model bundle.

    Expected keys in the joblib bundle:
        scaler
        model_full
        model_pruned
        feature_names
        pruned_feature_names

    Feature-group weighting is intentionally NOT applied.
    """
    obj = joblib.load(model_file)

    scaler = obj["scaler"]
    model_full = obj["model_full"]
    model_pruned = obj["model_pruned"]

    feature_names = list(obj["feature_names"])
    pruned_feature_names = list(
        obj["pruned_feature_names"]
    )

    X_test = np.load(X_test_file)
    y_test = np.load(y_test_file).reshape(-1)

    X_scaled = scaler.transform(X_test)

    full_pred = model_full.predict(
        X_scaled
    )

    name_to_index = {
        name: i
        for i, name in enumerate(feature_names)
    }

    pruned_indices = []

    for name in pruned_feature_names:
        if name not in name_to_index:
            raise ValueError(
                f"Pruned feature not found in full feature list: {name}"
            )
        pruned_indices.append(
            name_to_index[name]
        )

    X_pruned = X_scaled[
        :,
        pruned_indices,
    ]

    pruned_pred = model_pruned.predict(
        X_pruned
    )

    return {
        "y_true": y_test,
        "full_pred": full_pred,
        "pruned_pred": pruned_pred,
        "feature_names": feature_names,
        "pruned_feature_names": pruned_feature_names,
    }


def run_mcnemar_from_saved_models(
    model_file,
    X_test_file,
    y_test_file,
    save_dir=None,
    alpha=0.05,
):
    """
    Reproduce Full-vs-Pruned independent-test predictions and McNemar analysis.
    """
    data = predict_from_saved_models(
        model_file=model_file,
        X_test_file=X_test_file,
        y_test_file=y_test_file,
    )

    y_true = data["y_true"]
    full_pred = data["full_pred"]
    pruned_pred = data["pruned_pred"]

    full_acc = accuracy_score(
        y_true,
        full_pred,
    )
    full_f1 = f1_score(
        y_true,
        full_pred,
        average="macro",
        zero_division=0,
    )

    pruned_acc = accuracy_score(
        y_true,
        pruned_pred,
    )
    pruned_f1 = f1_score(
        y_true,
        pruned_pred,
        average="macro",
        zero_division=0,
    )

    result = calculate_mcnemar(
        y_true,
        full_pred,
        pruned_pred,
    )

    result.update(
        {
            "Full features": len(
                data["feature_names"]
            ),
            "Pruned features": len(
                data["pruned_feature_names"]
            ),
            "Full Accuracy (%)": full_acc * 100.0,
            "Full Macro-F1 (%)": full_f1 * 100.0,
            "Pruned Accuracy (%)": pruned_acc * 100.0,
            "Pruned Macro-F1 (%)": pruned_f1 * 100.0,
            "alpha": float(alpha),
            "statistically_significant_exact": (
                result["Exact McNemar p-value"]
                < alpha
            ),
        }
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        full_correct = (
            full_pred == y_true
        )
        pruned_correct = (
            pruned_pred == y_true
        )

        pd.DataFrame(
            {
                "sample_index": np.arange(
                    len(y_true)
                ),
                "y_true": y_true,
                "full_pred": full_pred,
                "pruned_pred": pruned_pred,
                "full_correct": full_correct.astype(int),
                "pruned_correct": pruned_correct.astype(int),
            }
        ).to_csv(
            save_dir / "Full_vs_Pruned_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            [result]
        ).to_csv(
            save_dir / "McNemar_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with (
            save_dir / "McNemar_report.txt"
        ).open("w", encoding="utf-8") as f:
            f.write(
                "Full SVM vs Pruned SVM - McNemar Test\n"
            )
            f.write("=" * 60 + "\n")
            for key, value in result.items():
                f.write(f"{key} = {value}\n")

    return result
