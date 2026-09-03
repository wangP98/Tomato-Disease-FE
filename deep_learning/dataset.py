# -*- coding: utf-8 -*-
"""Dataset helpers for CNN comparison experiments."""

from collections import Counter
from pathlib import Path
import json
import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)
from torchvision import transforms


DATASET_MEAN = [0.368, 0.381, 0.343]
DATASET_STD = [0.203, 0.185, 0.184]

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
}


def load_class_mapping(
    class_map_path="datasets/class_to_id.json",
):
    with Path(class_map_path).open(
        "r",
        encoding="utf-8",
    ) as f:
        class_to_id = json.load(f)["class_to_id"]

    id_to_class = {
        v: k
        for k, v in class_to_id.items()
    }

    return class_to_id, id_to_class


class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir,
        class_map_path="datasets/class_to_id.json",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        (
            self.class_to_idx,
            self.idx_to_class,
        ) = load_class_mapping(
            class_map_path
        )

        self.class_names = [
            name
            for name, _ in sorted(
                self.class_to_idx.items(),
                key=lambda kv: kv[1],
            )
        ]

        self.image_paths = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        for class_name in self.class_names:
            class_dir = (
                self.root_dir
                / class_name
            )

            if not class_dir.is_dir():
                continue

            for path in sorted(
                class_dir.iterdir(),
                key=lambda p: p.name.casefold(),
            ):
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTS
                ):
                    self.image_paths.append(path)
                    self.labels.append(
                        self.class_to_idx[class_name]
                    )

    def __len__(self):
        return len(
            self.image_paths
        )

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(
            path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(
                image
            )

        return image, label

    def get_class_weights(self):
        counts = Counter(
            self.labels
        )

        n = len(
            self.labels
        )

        c = len(
            self.class_to_idx
        )

        values = [
            n / (
                c
                * counts.get(
                    class_id,
                    1,
                )
            )
            for class_id in range(c)
        ]

        return torch.tensor(
            values,
            dtype=torch.float32,
        )


def build_train_transform(
    image_size=224,
):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(
                image_size
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DATASET_MEAN,
                std=DATASET_STD,
            ),
        ]
    )


def build_eval_transform(
    image_size=224,
):
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DATASET_MEAN,
                std=DATASET_STD,
            ),
        ]
    )


def create_data_loader(
    dataset,
    batch_size=4,
    train=False,
    random_seed=42,
):
    use_cuda = torch.cuda.is_available()

    num_workers = min(
        8,
        os.cpu_count() or 4,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            random_seed
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(
            num_workers > 0
        ),
        generator=(
            generator
            if train
            else None
        ),
    )


def build_weighted_sampler(
    dataset,
):
    counts = Counter(
        dataset.labels
    )

    n = len(
        dataset.labels
    )

    c = len(
        dataset.class_to_idx
    )

    class_weights = torch.tensor(
        [
            n
            / (
                c
                * counts.get(
                    i,
                    1,
                )
            )
            for i in range(c)
        ],
        dtype=torch.float32,
    )

    sample_weights = torch.tensor(
        [
            class_weights[y]
            for y in dataset.labels
        ],
        dtype=torch.float32,
    )

    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )
