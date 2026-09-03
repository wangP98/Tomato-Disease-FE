# -*- coding: utf-8 -*-
"""
Unified training script for four pretrained CNN baselines:

1. AlexNet
2. VGG-16
3. EfficientNet-B0
4. MobileNetV3-Large

The training protocol follows the supplied AlexNet example, but fixes one
important reproducibility issue: the independent test set is NOT used for
scheduler updates, checkpoint selection, or any other training decision.

Instead:
    original TRAIN set
        -> stratified train/validation split
        -> training + validation-based model selection
        -> load best validation checkpoint
        -> evaluate TEST once

All four models use the same:
- dataset split
- image size
- data augmentation
- optimizer family
- learning-rate scheduler
- selection criterion
- class mapping

No feature-group weighting is involved.

Example
-------
Train one model:

python deep_learning/train_cnn_baselines.py \
    --model alexnet \
    --data-root data01 \
    --class-map datasets/class_to_id.json

Train all four sequentially:

python deep_learning/train_cnn_baselines.py \
    --model all \
    --data-root data01 \
    --class-map datasets/class_to_id.json
"""

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# ============================================================
# Default configuration
# ============================================================

RANDOM_STATE = 42

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
MOMENTUM = 0.9

VAL_RATIO = 0.10
NUM_WORKERS = 4

SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_CHOICES = [
    "alexnet",
    "vgg16",
    "efficientnet_b0",
    "mobilenet_v3_large",
]

MODEL_DISPLAY_NAMES = {
    "alexnet": "AlexNet",
    "vgg16": "VGG-16",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNetV3-Large",
}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducible behavior.
    # This can reduce speed slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Class mapping
# ============================================================

def load_class_mapping(class_map_path):
    class_map_path = Path(class_map_path)

    with class_map_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        class_to_id = json.load(f)["class_to_id"]

    id_to_class = {
        int(class_id): class_name
        for class_name, class_id
        in class_to_id.items()
    }

    class_names = [
        id_to_class[i]
        for i in sorted(id_to_class)
    ]

    return class_to_id, id_to_class, class_names


# ============================================================
# Dataset
# ============================================================

class CanonicalImageFolderDataset(
    torch.utils.data.Dataset
):
    """
    Image-folder dataset that uses datasets/class_to_id.json rather than
    torchvision ImageFolder's alphabetical class ordering.

    Expected structure:
        root/
            Tomato__Target_Spot/
            Tomato__Tomato_mosaic_virus/
            ...
    """

    IMAGE_EXTS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".webp",
    }

    def __init__(
        self,
        root_dir,
        class_to_id,
        transform=None,
    ):
        from PIL import Image

        self.Image = Image
        self.root_dir = Path(root_dir)
        self.class_to_id = dict(class_to_id)
        self.transform = transform

        self.samples = []

        ordered_classes = sorted(
            self.class_to_id.items(),
            key=lambda kv: kv[1],
        )

        for class_name, class_id in ordered_classes:
            class_dir = (
                self.root_dir
                / class_name
            )

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            image_paths = sorted(
                [
                    p
                    for p in class_dir.rglob("*")
                    if p.is_file()
                    and p.suffix.lower()
                    in self.IMAGE_EXTS
                ],
                key=lambda p: p.as_posix(),
            )

            for image_path in image_paths:
                self.samples.append(
                    (
                        image_path,
                        int(class_id),
                    )
                )

        self.labels = [
            label
            for _, label in self.samples
        ]

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(self, index):
        image_path, label = (
            self.samples[index]
        )

        image = (
            self.Image
            .open(image_path)
            .convert("RGB")
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        return image, label


# ============================================================
# Transforms
# ============================================================

def build_transforms(
    image_size=IMAGE_SIZE,
):
    """
    Training augmentation follows the supplied AlexNet example.
    Validation/test use deterministic preprocessing only.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                15
            ),

            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return (
        train_transform,
        eval_transform,
    )


# ============================================================
# DataLoaders
# ============================================================

def build_dataloaders(
    data_root,
    class_to_id,
    batch_size=BATCH_SIZE,
    val_ratio=VAL_RATIO,
    num_workers=NUM_WORKERS,
    random_state=RANDOM_STATE,
):
    """
    Create:
        train_loader
        val_loader
        test_loader

    Validation is split only from the original training set.
    The independent test set remains untouched until final evaluation.
    """

    data_root = Path(data_root)

    train_dir = (
        data_root
        / "train"
    )

    test_dir = (
        data_root
        / "test"
    )

    (
        train_transform,
        eval_transform,
    ) = build_transforms()

    # Same file ordering, different transforms.
    train_aug_dataset = (
        CanonicalImageFolderDataset(
            train_dir,
            class_to_id,
            transform=train_transform,
        )
    )

    train_eval_dataset = (
        CanonicalImageFolderDataset(
            train_dir,
            class_to_id,
            transform=eval_transform,
        )
    )

    test_dataset = (
        CanonicalImageFolderDataset(
            test_dir,
            class_to_id,
            transform=eval_transform,
        )
    )

    indices = np.arange(
        len(train_aug_dataset)
    )

    labels = np.asarray(
        train_aug_dataset.labels,
        dtype=np.int64,
    )

    train_idx, val_idx = (
        train_test_split(
            indices,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=labels,
        )
    )

    train_subset = Subset(
        train_aug_dataset,
        train_idx,
    )

    val_subset = Subset(
        train_eval_dataset,
        val_idx,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            random_state
        )
    )

    pin_memory = (
        torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_aug_dataset,
        "test_dataset": test_dataset,
        "train_idx": np.asarray(train_idx),
        "val_idx": np.asarray(val_idx),
    }


# ============================================================
# Model construction
# ============================================================

def build_model(
    model_name,
    num_classes,
    pretrained=True,
    fine_tune_mode="all",
):
    """
    Build a torchvision CNN and replace its classification head.

    fine_tune_mode:
        "all"        -> all parameters are trainable
        "classifier" -> freeze feature extractor and train only classifier

    The default is full fine-tuning because it avoids imposing arbitrary
    architecture-specific partial-freezing rules across the four models.
    """

    if model_name == "alexnet":

        weights = (
            models.AlexNet_Weights
            .IMAGENET1K_V1
            if pretrained
            else None
        )

        model = models.alexnet(
            weights=weights
        )

        in_features = (
            model.classifier[6]
            .in_features
        )

        model.classifier[6] = (
            nn.Linear(
                in_features,
                num_classes,
            )
        )

        classifier_module = (
            model.classifier
        )


    elif model_name == "vgg16":

        weights = (
            models.VGG16_Weights
            .IMAGENET1K_V1
            if pretrained
            else None
        )

        model = models.vgg16(
            weights=weights
        )

        in_features = (
            model.classifier[6]
            .in_features
        )

        model.classifier[6] = (
            nn.Linear(
                in_features,
                num_classes,
            )
        )

        classifier_module = (
            model.classifier
        )


    elif model_name == "efficientnet_b0":

        weights = (
            models.EfficientNet_B0_Weights
            .IMAGENET1K_V1
            if pretrained
            else None
        )

        model = models.efficientnet_b0(
            weights=weights
        )

        in_features = (
            model.classifier[1]
            .in_features
        )

        model.classifier[1] = (
            nn.Linear(
                in_features,
                num_classes,
            )
        )

        classifier_module = (
            model.classifier
        )


    elif model_name == "mobilenet_v3_large":

        weights = (
            models.MobileNet_V3_Large_Weights
            .IMAGENET1K_V2
            if pretrained
            else None
        )

        model = (
            models.mobilenet_v3_large(
                weights=weights
            )
        )

        in_features = (
            model.classifier[3]
            .in_features
        )

        model.classifier[3] = (
            nn.Linear(
                in_features,
                num_classes,
            )
        )

        classifier_module = (
            model.classifier
        )


    else:

        raise ValueError(
            f"Unsupported model: "
            f"{model_name}"
        )


    if fine_tune_mode == "classifier":

        for parameter in (
            model.parameters()
        ):
            parameter.requires_grad = False

        for parameter in (
            classifier_module.parameters()
        ):
            parameter.requires_grad = True


    elif fine_tune_mode == "all":

        for parameter in (
            model.parameters()
        ):
            parameter.requires_grad = True


    else:

        raise ValueError(
            "fine_tune_mode must be "
            "'all' or 'classifier'."
        )

    return model


# ============================================================
# Epoch pass
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    is_training = (
        optimizer is not None
    )

    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    n_samples = 0

    all_true = []
    all_pred = []

    for inputs, labels in loader:

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                labels,
            )

            preds = (
                outputs.argmax(
                    dim=1
                )
            )

            if is_training:

                loss.backward()

                optimizer.step()

        batch_size = inputs.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        running_correct += (
            preds.eq(labels)
            .sum()
            .item()
        )

        n_samples += batch_size

        all_true.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            preds.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    loss_mean = (
        running_loss
        / max(
            n_samples,
            1,
        )
    )

    accuracy = (
        running_correct
        / max(
            n_samples,
            1,
        )
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_true,
            all_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "loss": float(loss_mean),
        "accuracy": float(accuracy),
        "precision_macro": float(
            precision
        ),
        "recall_macro": float(
            recall
        ),
        "f1_macro": float(
            f1
        ),
    }


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_dir,
):
    """
    Checkpoint selection is based on validation accuracy.
    Test data are never accessed here.
    """

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    best_val_acc = -1.0
    best_epoch = -1

    history = []

    start_time = time.time()

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Epoch "
            f"{epoch}/{num_epochs}"
        )

        print(
            "=" * 70
        )

        train_metrics = run_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_precision_macro":
                train_metrics[
                    "precision_macro"
                ],

            "train_recall_macro":
                train_metrics[
                    "recall_macro"
                ],

            "train_f1_macro":
                train_metrics[
                    "f1_macro"
                ],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_precision_macro":
                val_metrics[
                    "precision_macro"
                ],

            "val_recall_macro":
                val_metrics[
                    "recall_macro"
                ],

            "val_f1_macro":
                val_metrics[
                    "f1_macro"
                ],
        }

        history.append(
            row
        )

        print(
            f"Train Loss : "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train Acc  : "
            f"{train_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Train F1   : "
            f"{train_metrics['f1_macro'] * 100:.2f}%"
        )

        print(
            f"Val Loss   : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Val F1     : "
            f"{val_metrics['f1_macro'] * 100:.2f}%"
        )

        print(
            f"LR         : "
            f"{current_lr:.8f}"
        )

        # Same selection criterion concept as the supplied AlexNet example,
        # but validation rather than test accuracy is used.
        if (
            val_metrics["accuracy"]
            > best_val_acc
        ):

            best_val_acc = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            best_weights = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            torch.save(
                best_weights,
                save_dir
                / "best_model.pth",
            )

            print(
                ">> New best validation "
                "checkpoint saved."
            )

    elapsed = (
        time.time()
        - start_time
    )

    model.load_state_dict(
        best_weights
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        save_dir
        / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        model,
        history_df,
        best_epoch,
        best_val_acc,
        elapsed,
    )


# ============================================================
# Test prediction
# ============================================================

@torch.no_grad()
def predict_loader(
    model,
    loader,
    device,
):
    model.eval()

    y_true = []
    y_pred = []

    for inputs, labels in loader:

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        outputs = model(
            inputs
        )

        preds = outputs.argmax(
            dim=1
        )

        y_true.extend(
            labels.numpy()
            .tolist()
        )

        y_pred.extend(
            preds.cpu()
            .numpy()
            .tolist()
        )

    return (
        np.asarray(
            y_true,
            dtype=np.int64,
        ),
        np.asarray(
            y_pred,
            dtype=np.int64,
        ),
    )


# ============================================================
# Metrics
# ============================================================

def save_performance_metrics(
    y_true,
    y_pred,
    class_names,
    save_dir,
    model_name,
):
    labels = list(
        range(
            len(class_names)
        )
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    p_cls, r_cls, f_cls, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Accuracy (%)":
                    accuracy * 100.0,
                "Macro Precision (%)":
                    precision * 100.0,
                "Macro Recall (%)":
                    recall * 100.0,
                "Macro F1 (%)":
                    f1 * 100.0,
            }
        ]
    )

    summary_df.to_csv(
        save_dir
        / "test_summary_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_df = pd.DataFrame(
        {
            "Class_ID": labels,
            "Class": class_names,
            "Precision (%)":
                p_cls * 100.0,
            "Recall (%)":
                r_cls * 100.0,
            "F1 (%)":
                f_cls * 100.0,
            "Support":
                support.astype(int),
        }
    )

    class_df.to_csv(
        save_dir
        / "test_class_wise_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        summary_df,
        class_df,
    )


# ============================================================
# Confusion matrix
# ============================================================

def save_confusion_matrices(
    y_true,
    y_pred,
    class_names,
    save_dir,
):
    labels = list(
        range(
            len(class_names)
        )
    )

    # Counts
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        save_dir
        / "test_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
        xticks_rotation=45,
    )

    ax.set_title(
        "Test Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        save_dir
        / "test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Row-normalized
    cm_norm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )

    pd.DataFrame(
        cm_norm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        save_dir
        / "test_confusion_matrix_normalized.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=class_names,
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        colorbar=True,
        xticks_rotation=45,
    )

    ax.set_title(
        "Test Confusion Matrix (Normalized)"
    )

    fig.tight_layout()

    fig.savefig(
        save_dir
        / "test_confusion_matrix_normalized.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Training curves
# ============================================================

def save_training_curves(
    history_df,
    save_dir,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train",
    )

    ax.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "Training and Validation Loss"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_dir
        / "loss_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history_df["epoch"],
        history_df["train_accuracy"]
        * 100.0,
        label="Train",
    )

    ax.plot(
        history_df["epoch"],
        history_df["val_accuracy"]
        * 100.0,
        label="Validation",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Training and Validation Accuracy"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_dir
        / "accuracy_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Split manifests
# ============================================================

def save_split_manifest(
    data_bundle,
    class_names,
    save_dir,
):
    dataset = (
        data_bundle[
            "train_dataset"
        ]
    )

    rows_train = []

    for idx in (
        data_bundle[
            "train_idx"
        ]
    ):
        path, label = (
            dataset.samples[
                int(idx)
            ]
        )

        rows_train.append(
            {
                "relative_path":
                    path.relative_to(
                        dataset.root_dir
                    ).as_posix(),
                "class_id":
                    int(label),
                "class_name":
                    class_names[
                        int(label)
                    ],
            }
        )

    rows_val = []

    for idx in (
        data_bundle[
            "val_idx"
        ]
    ):
        path, label = (
            dataset.samples[
                int(idx)
            ]
        )

        rows_val.append(
            {
                "relative_path":
                    path.relative_to(
                        dataset.root_dir
                    ).as_posix(),
                "class_id":
                    int(label),
                "class_name":
                    class_names[
                        int(label)
                    ],
            }
        )

    pd.DataFrame(
        rows_train
    ).to_csv(
        save_dir
        / "cnn_train_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        rows_val
    ).to_csv(
        save_dir
        / "cnn_validation_split.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Train one CNN
# ============================================================

def train_one_model(
    model_name,
    data_root,
    class_map_path,
    output_root,
    batch_size,
    epochs,
    learning_rate,
    val_ratio,
    num_workers,
    fine_tune_mode,
    pretrained,
):
    seed_everything(
        RANDOM_STATE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        class_to_id,
        id_to_class,
        class_names,
    ) = load_class_mapping(
        class_map_path
    )

    data_bundle = build_dataloaders(
        data_root=data_root,
        class_to_id=class_to_id,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers,
        random_state=RANDOM_STATE,
    )

    model = build_model(
        model_name=model_name,
        num_classes=len(
            class_names
        ),
        pretrained=pretrained,
        fine_tune_mode=fine_tune_mode,
    )

    model = model.to(
        device
    )

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = optim.SGD(
        trainable_parameters,
        lr=learning_rate,
        momentum=MOMENTUM,
    )

    scheduler = (
        optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    display_name = (
        MODEL_DISPLAY_NAMES[
            model_name
        ]
    )

    save_dir = (
        Path(output_root)
        / model_name
        / timestamp
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "#" * 80
    )

    print(
        f"MODEL: {display_name}"
    )

    print(
        f"DEVICE: {device}"
    )

    print(
        f"TRAIN samples: "
        f"{len(data_bundle['train'].dataset)}"
    )

    print(
        f"VAL samples  : "
        f"{len(data_bundle['val'].dataset)}"
    )

    print(
        f"TEST samples : "
        f"{len(data_bundle['test'].dataset)}"
    )

    print(
        "#" * 80
    )

    save_split_manifest(
        data_bundle,
        class_names,
        save_dir,
    )

    config = {
        "model": display_name,
        "architecture_key": model_name,

        "pretrained_imagenet":
            bool(pretrained),

        "fine_tune_mode":
            fine_tune_mode,

        "random_state":
            RANDOM_STATE,

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            batch_size,

        "epochs":
            epochs,

        "learning_rate":
            learning_rate,

        "optimizer":
            "SGD",

        "momentum":
            MOMENTUM,

        "scheduler":
            "ReduceLROnPlateau",

        "scheduler_factor":
            SCHEDULER_FACTOR,

        "scheduler_patience":
            SCHEDULER_PATIENCE,

        "validation_ratio_from_training":
            val_ratio,

        "checkpoint_selection":
            "highest validation accuracy",

        "test_used_during_training":
            False,

        "normalization_mean":
            IMAGENET_MEAN,

        "normalization_std":
            IMAGENET_STD,

        "augmentation": {
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.5,
            "rotation_degrees": 15,
            "color_jitter_brightness": 0.1,
            "color_jitter_contrast": 0.1,
            "color_jitter_saturation": 0.1,
        },

        "class_to_id":
            class_to_id,
    }

    with (
        save_dir
        / "training_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    (
        model,
        history_df,
        best_epoch,
        best_val_acc,
        elapsed,
    ) = train_model(
        model=model,
        dataloaders=data_bundle,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=epochs,
        save_dir=save_dir,
    )

    save_training_curves(
        history_df,
        save_dir,
    )

    # ========================================================
    # Independent TEST evaluation: only after training ends
    # ========================================================

    print(
        "\n>> Final independent "
        "TEST evaluation..."
    )

    (
        y_true,
        y_pred,
    ) = predict_loader(
        model,
        data_bundle["test"],
        device,
    )

    (
        summary_df,
        class_df,
    ) = save_performance_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_dir=save_dir,
        model_name=display_name,
    )

    save_confusion_matrices(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_dir=save_dir,
    )

    predictions_df = pd.DataFrame(
        {
            "sample_index":
                np.arange(
                    len(y_true)
                ),

            "y_true":
                y_true,

            "true_class":
                [
                    class_names[i]
                    for i in y_true
                ],

            "y_pred":
                y_pred,

            "predicted_class":
                [
                    class_names[i]
                    for i in y_pred
                ],

            "correct":
                (
                    y_true
                    == y_pred
                ).astype(int),
        }
    )

    predictions_df.to_csv(
        save_dir
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "best_epoch":
            int(best_epoch),

        "best_validation_accuracy":
            float(best_val_acc),

        "training_seconds":
            float(elapsed),

        "test_accuracy_percent":
            float(
                summary_df.loc[
                    0,
                    "Accuracy (%)",
                ]
            ),

        "test_macro_precision_percent":
            float(
                summary_df.loc[
                    0,
                    "Macro Precision (%)",
                ]
            ),

        "test_macro_recall_percent":
            float(
                summary_df.loc[
                    0,
                    "Macro Recall (%)",
                ]
            ),

        "test_macro_f1_percent":
            float(
                summary_df.loc[
                    0,
                    "Macro F1 (%)",
                ]
            ),
    }

    with (
        save_dir
        / "run_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save checkpoint with enough metadata for later benchmarking.
    torch.save(
        {
            "architecture":
                model_name,

            "display_name":
                display_name,

            "model_state_dict":
                model.state_dict(),

            "class_to_id":
                class_to_id,

            "num_classes":
                len(class_names),

            "image_size":
                IMAGE_SIZE,

            "normalization_mean":
                IMAGENET_MEAN,

            "normalization_std":
                IMAGENET_STD,

            "best_epoch":
                best_epoch,

            "best_validation_accuracy":
                best_val_acc,
        },
        save_dir
        / "best_checkpoint.pth",
    )

    print(
        "\nTEST RESULTS"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        save_dir
    )

    return {
        "model": display_name,
        "save_dir": str(
            save_dir
        ),
        **metadata,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "all",
            *MODEL_CHOICES,
        ],
        default="all",
    )

    parser.add_argument(
        "--data-root",
        default="data01",
    )

    parser.add_argument(
        "--class-map",
        default=(
            "datasets/"
            "class_to_id.json"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/"
            "deep_learning"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
    )

    parser.add_argument(
        "--fine-tune-mode",
        choices=[
            "all",
            "classifier",
        ],
        default="all",
        help=(
            "'all': fine-tune the complete network; "
            "'classifier': train only the classification head."
        ),
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Disable ImageNet pretrained weights."
        ),
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.val_ratio
        < 1.0
    ):
        raise ValueError(
            "--val-ratio must "
            "be between 0 and 1."
        )

    models_to_run = (
        MODEL_CHOICES
        if args.model == "all"
        else [args.model]
    )

    all_results = []

    for model_name in models_to_run:

        result = train_one_model(
            model_name=model_name,
            data_root=args.data_root,
            class_map_path=args.class_map,
            output_root=args.output_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            fine_tune_mode=args.fine_tune_mode,
            pretrained=(
                not args.no_pretrained
            ),
        )

        all_results.append(
            result
        )

    if len(all_results) > 1:

        summary_path = (
            Path(args.output_root)
            / "cnn_training_summary.csv"
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            all_results
        ).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\nCombined summary:"
        )

        print(
            summary_path
        )


if __name__ == "__main__":
    main()
