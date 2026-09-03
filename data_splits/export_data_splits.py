# -*- coding: utf-8 -*-
"""
Export dataset relative paths to text files.

Expected dataset structure
--------------------------
data_root/
├── train/
│   ├── class_A/
│   ├── class_B/
│   └── ...
└── test/
    ├── class_A/
    ├── class_B/
    └── ...

Each output TXT contains one relative image path per line, relative to
``data_root``. The class label is therefore preserved in the path itself.

Optionally, deterministic stratified five-fold train/validation path files can
also be generated from the training split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_STATE = 42
N_SPLITS = 5


def load_class_to_id(class_map_path):
    """Load {'class_to_id': {...}} from JSON."""
    with Path(class_map_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["class_to_id"]


def collect_split_paths(
    data_root,
    split_name,
    class_to_id,
):
    """
    Collect image paths in deterministic order.

    Ordering:
    1. class_to_id order by numeric class ID;
    2. lexicographically sorted relative path within each class.
    """
    data_root = Path(data_root)
    split_dir = data_root / split_name

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}"
        )

    items = sorted(
        class_to_id.items(),
        key=lambda kv: kv[1],
    )

    rows = []

    for class_name, class_id in items:
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        image_paths = sorted(
            [
                p
                for p in class_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMG_EXTS
            ],
            key=lambda p: p.as_posix(),
        )

        for path in image_paths:
            rel = path.relative_to(data_root).as_posix()

            rows.append(
                {
                    "relative_path": rel,
                    "split": split_name,
                    "class_name": class_name,
                    "class_id": int(class_id),
                }
            )

    return pd.DataFrame(rows)


def write_path_txt(df, output_path):
    """Write one relative path per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for path in df["relative_path"]:
            f.write(f"{path}\n")


def export_train_test(
    data_root,
    class_map_path,
    output_dir,
):
    """Export train.txt, test.txt, and a manifest CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_to_id = load_class_to_id(
        class_map_path
    )

    train_df = collect_split_paths(
        data_root,
        "train",
        class_to_id,
    )

    test_df = collect_split_paths(
        data_root,
        "test",
        class_to_id,
    )

    write_path_txt(
        train_df,
        output_dir / "train.txt",
    )

    write_path_txt(
        test_df,
        output_dir / "test.txt",
    )

    manifest = pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )

    manifest.to_csv(
        output_dir / "dataset_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        manifest
        .groupby(
            ["split", "class_id", "class_name"]
        )
        .size()
        .reset_index(name="n_images")
    )

    summary.to_csv(
        output_dir / "split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return train_df, test_df


def export_stratified_folds(
    train_df,
    output_dir,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Generate deterministic stratified fold path lists from train_df.

    Important
    ---------
    These folds are regenerated from the deterministic path ordering used in
    this script. If an older experiment used a different sample ordering,
    its exact historical fold membership should be exported from that original
    run instead of being assumed to match these regenerated folds.
    """
    output_dir = Path(output_dir) / "folds"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = train_df[
        "relative_path"
    ].to_numpy()

    labels = train_df[
        "class_id"
    ].to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(paths, labels),
        start=1,
    ):
        fold_train = train_df.iloc[
            train_idx
        ].reset_index(drop=True)

        fold_val = train_df.iloc[
            val_idx
        ].reset_index(drop=True)

        write_path_txt(
            fold_train,
            output_dir /
            f"fold{fold}_train.txt",
        )

        write_path_txt(
            fold_val,
            output_dir /
            f"fold{fold}_val.txt",
        )

        fold_rows.append(
            {
                "fold": fold,
                "train_images": len(fold_train),
                "validation_images": len(fold_val),
            }
        )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        output_dir / "fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export train/test relative paths and "
            "optional stratified five-fold path lists."
        )
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Dataset root containing train/ and test/."
        ),
    )

    parser.add_argument(
        "--class-map",
        required=True,
        help=(
            "JSON file containing {'class_to_id': {...}}."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="data_splits",
    )

    parser.add_argument(
        "--make-folds",
        action="store_true",
        help=(
            "Also generate 5-fold train/validation "
            "relative-path TXT files."
        ),
    )

    args = parser.parse_args()

    train_df, _ = export_train_test(
        data_root=args.data_root,
        class_map_path=args.class_map,
        output_dir=args.output_dir,
    )

    if args.make_folds:
        export_stratified_folds(
            train_df=train_df,
            output_dir=args.output_dir,
        )

    print(
        f"Split files saved to: "
        f"{Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
