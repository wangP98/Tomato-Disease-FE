# -*- coding: utf-8 -*-
"""
Generate a deterministic class_to_id.json from class subdirectories.

The existing datasets/class_to_id.json should be treated as the canonical
mapping for reproducing the reported experiments. This utility is provided
only for rebuilding a mapping for another dataset.
"""

import argparse
import json
from pathlib import Path


def generate_class_mapping(data_dir, output_path):
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    classes = sorted(
        [p.name for p in data_dir.iterdir() if p.is_dir()],
        key=str.casefold,
    )

    class_to_id = {
        class_name: i
        for i, class_name in enumerate(classes)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"class_to_id": class_to_id},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved class mapping to: {output_path}")
    return class_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory whose immediate subfolders are class names.",
    )
    parser.add_argument(
        "--output",
        default="datasets/class_to_id.json",
    )
    args = parser.parse_args()

    generate_class_mapping(
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
