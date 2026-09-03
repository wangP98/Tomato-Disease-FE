# -*- coding: utf-8 -*-
"""
Create deterministic per-class train/test or train/val/test splits.

For exact reproduction of an already-created historical split, do not rerun
this utility. Instead use data_splits/export_data_splits.py on the existing
dataset directories and publish the resulting relative-path manifests.
"""

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
}


def load_class_names(class_map_path):
    with Path(class_map_path).open("r", encoding="utf-8") as f:
        mapping = json.load(f)["class_to_id"]

    return [
        name
        for name, _ in sorted(
            mapping.items(),
            key=lambda kv: kv[1],
        )
    ]


def list_images(class_dir):
    return sorted(
        [
            p
            for p in Path(class_dir).iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda p: p.name.casefold(),
    )


def copy_images(paths, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for src in paths:
        shutil.copy2(
            src,
            destination / src.name,
        )


def split_train_test(
    source_dir,
    output_dir,
    class_map_path,
    test_size=0.2,
    seed=42,
):
    """Per-class deterministic train/test split."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    class_names = load_class_names(class_map_path)

    for class_name in class_names:
        images = list_images(
            source_dir / class_name
        )

        rng.shuffle(images)

        test_count = int(
            len(images) * test_size
        )

        test_images = images[:test_count]
        train_images = images[test_count:]

        copy_images(
            train_images,
            output_dir / "train" / class_name,
        )
        copy_images(
            test_images,
            output_dir / "test" / class_name,
        )

        print(
            f"{class_name}: "
            f"train={len(train_images)}, "
            f"test={len(test_images)}"
        )


def split_train_val_test(
    source_dir,
    output_dir,
    class_map_path,
    test_size=0.1,
    val_size=0.1,
    seed=42,
):
    """Per-class deterministic train/validation/test split."""
    if test_size < 0 or val_size < 0 or test_size + val_size >= 1:
        raise ValueError(
            "Require test_size >= 0, val_size >= 0, "
            "and test_size + val_size < 1."
        )

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    class_names = load_class_names(class_map_path)

    for class_name in class_names:
        images = list_images(
            source_dir / class_name
        )
        rng.shuffle(images)

        n = len(images)
        n_test = int(n * test_size)
        n_val = int(n * val_size)

        test_images = images[:n_test]
        val_images = images[n_test:n_test + n_val]
        train_images = images[n_test + n_val:]

        copy_images(
            train_images,
            output_dir / "train" / class_name,
        )
        copy_images(
            val_images,
            output_dir / "val" / class_name,
        )
        copy_images(
            test_images,
            output_dir / "test" / class_name,
        )

        print(
            f"{class_name}: "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--class-map",
        default="datasets/class_to_id.json",
    )
    parser.add_argument(
        "--mode",
        choices=["train_test", "train_val_test"],
        default="train_test",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.mode == "train_test":
        split_train_test(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            class_map_path=args.class_map,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        split_train_val_test(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            class_map_path=args.class_map,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
