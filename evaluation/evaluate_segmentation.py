# -*- coding: utf-8 -*-
"""
Evaluate predicted leaf masks against reference masks.

Metrics
-------
Dice
IoU
Precision
Recall

Masks are matched by relative path. Each predicted/reference mask is binarized
using >0 as foreground.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def binary_mask(path):
    """Read a mask and convert it to a boolean array."""
    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not read mask: {path}"
        )

    return mask > 0


def segmentation_metrics(
    reference,
    prediction,
):
    """Calculate Dice, IoU, Precision, and Recall."""
    reference = np.asarray(
        reference,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    if reference.shape != prediction.shape:
        raise ValueError(
            f"Mask shape mismatch: "
            f"{reference.shape} vs "
            f"{prediction.shape}"
        )

    tp = int(
        np.sum(
            reference & prediction
        )
    )

    fp = int(
        np.sum(
            (~reference) & prediction
        )
    )

    fn = int(
        np.sum(
            reference & (~prediction)
        )
    )

    dice = (
        2.0 * tp
        / (2.0 * tp + fp + fn + 1e-12)
    )

    iou = (
        tp
        / (tp + fp + fn + 1e-12)
    )

    precision = (
        tp
        / (tp + fp + 1e-12)
    )

    recall = (
        tp
        / (tp + fn + 1e-12)
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def collect_masks(root):
    """Return {relative_path: absolute_path} for all mask images."""
    root = Path(root)

    files = {}

    for p in root.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in IMG_EXTS
        ):
            rel = p.relative_to(
                root
            ).as_posix()

            files[rel] = p

    return files


def evaluate_directory(
    reference_root,
    prediction_root,
    output_dir,
):
    """Evaluate all prediction masks that match reference relative paths."""
    reference_root = Path(
        reference_root
    )

    prediction_root = Path(
        prediction_root
    )

    refs = collect_masks(
        reference_root
    )

    preds = collect_masks(
        prediction_root
    )

    common = sorted(
        set(refs)
        & set(preds)
    )

    if not common:
        raise RuntimeError(
            "No matching relative mask paths were found."
        )

    rows = []

    for rel in common:
        ref = binary_mask(
            refs[rel]
        )

        pred = binary_mask(
            preds[rel]
        )

        metrics = segmentation_metrics(
            ref,
            pred,
        )

        rows.append(
            {
                "relative_path": rel,
                **metrics,
            }
        )

    df = pd.DataFrame(
        rows
    )

    summary_rows = []

    for metric in [
        "dice",
        "iou",
        "precision",
        "recall",
    ]:
        mean_value = float(
            df[metric].mean()
        )

        std_value = float(
            df[metric].std(ddof=1)
        )

        summary_rows.append(
            {
                "metric": metric,
                "mean": mean_value,
                "std": std_value,
                "mean_percent": mean_value * 100.0,
                "std_percent": std_value * 100.0,
                "paper": (
                    f"{mean_value * 100:.2f} "
                    f"± "
                    f"{std_value * 100:.2f}"
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_dir /
        "segmentation_metrics_per_image.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        output_dir /
        "segmentation_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_pred = sorted(
        set(refs)
        - set(preds)
    )

    missing_ref = sorted(
        set(preds)
        - set(refs)
    )

    with (
        output_dir /
        "matching_report.txt"
    ).open("w", encoding="utf-8") as f:
        f.write(
            f"Matched masks: {len(common)}\n"
        )

        f.write(
            f"Reference masks without prediction: "
            f"{len(missing_pred)}\n"
        )

        f.write(
            f"Predictions without reference: "
            f"{len(missing_ref)}\n"
        )

    return df, summary_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference-root",
        required=True,
    )

    parser.add_argument(
        "--prediction-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="results/segmentation",
    )

    args = parser.parse_args()

    _, summary = evaluate_directory(
        reference_root=args.reference_root,
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
