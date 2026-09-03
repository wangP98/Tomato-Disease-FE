# -*- coding: utf-8 -*-
"""SVM evaluation, class-wise metrics, and confusion-matrix output."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred):
    """Return Accuracy and macro Precision/Recall/F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    class_to_id,
    title_prefix,
    normalize=None,
    output_dir="results",
    dpi=300,
    cmap="Blues",
    tick_fontsize=10,
    number_fontsize=12,
    rotate_xticks=45,
):
    """
    Save a confusion matrix with the same plotting logic used in the original
    evaluation script.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    items = sorted(class_to_id.items(), key=lambda kv: kv[1])
    display_names = [name for name, _ in items]
    label_ids = [class_id for _, class_id in items]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_ids,
        normalize=normalize,
    )

    accuracy = accuracy_score(y_true, y_pred) * 100.0

    n = len(label_ids)
    fig_w = max(12, 0.7 * n)
    fig_h = max(10, 0.7 * n)

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h),
        dpi=dpi,
    )

    fmt = ".2f" if normalize else "d"

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_names,
    )
    display.plot(
        ax=ax,
        cmap=cmap,
        values_format=fmt,
        colorbar=True,
    )

    norm_str = f" (normalize={normalize})" if normalize else ""
    ax.set_title(
        f"{title_prefix} Confusion Matrix "
        f"(Accuracy: {accuracy:.2f}%)"
        f"{norm_str}",
        fontsize=16,
        pad=12,
    )
    ax.set_xlabel("Predicted label", fontsize=12, labelpad=8)
    ax.set_ylabel("True label", fontsize=12, labelpad=8)

    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    for label in ax.get_xticklabels():
        label.set_rotation(rotate_xticks)
        label.set_ha("right")

    if hasattr(display, "text_"):
        image = display.im_
        vmin, vmax = image.get_clim()
        threshold = (vmin + vmax) / 2.0

        texts = display.text_
        text_iter = texts.flat if isinstance(texts, np.ndarray) else texts

        for text in text_iter:
            if hasattr(text, "set_fontsize"):
                text.set_fontsize(number_fontsize)
                try:
                    value = float(text.get_text())
                except ValueError:
                    value = 0.0
                text.set_color(
                    "white" if value > threshold else "black"
                )

    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_norm-{normalize}" if normalize else ""
    png_path = output_dir / f"{title_prefix}_cm{suffix}.png"
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save the numerical matrix as CSV as well.
    csv_path = output_dir / f"{title_prefix}_cm{suffix}.csv"
    pd.DataFrame(
        cm,
        index=display_names,
        columns=display_names,
    ).to_csv(csv_path, encoding="utf-8-sig")

    return cm


def evaluate_model(
    model,
    X,
    y,
    class_to_id,
    prefix,
    output_dir="results",
):
    """Evaluate a fitted model and save numerical/class-wise outputs."""
    y_true = np.asarray(y).ravel()
    y_pred = np.asarray(model.predict(X)).ravel()

    items = sorted(class_to_id.items(), key=lambda kv: kv[1])
    label_ids = [class_id for _, class_id in items]
    display_names = [name for name, _ in items]

    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=display_names,
        zero_division=0,
        output_dict=True,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(report).T.to_csv(
        output_dir / f"{prefix}_classification_report.csv",
        encoding="utf-8-sig",
    )

    metrics = compute_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(
        output_dir / f"{prefix}_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        class_to_id,
        prefix,
        normalize=None,
        output_dir=output_dir,
    )
    save_confusion_matrix(
        y_true,
        y_pred,
        class_to_id,
        prefix,
        normalize="true",
        output_dir=output_dir,
    )

    return y_pred, metrics, report
