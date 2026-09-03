# -*- coding: utf-8 -*-

"""
Reviewer 2 - Comment 7
Efficiency benchmark for four deep-learning classifiers:

1. AlexNet
2. VGG-16
3. EfficientNet-B0
4. MobileNetV3-Large

Metrics:
- Parameters (M)
- Checkpoint size (MB)
- Pure forward latency (ms/image)
- Pure forward FPS
- Peak GPU memory (MB)
- End-to-end latency (image loading + preprocessing + H2D + inference)
- End-to-end FPS
- Test accuracy

Important:
- batch size = 1
- FP32
- CUDA synchronize before/after timing
- same 224x224 input
- no training
"""

import os
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data01" / "test"

MODEL_PATHS = {
    "AlexNet": ROOT / "checkpoints" / "AlexNet" / "best_model.pth",
    "VGG-16": ROOT / "checkpoints" / "VGG16" / "best_model.pth",
    "EfficientNet-B0": ROOT / "checkpoints" / "EfficientNet-B0" / "best_model.pth",
    "MobileNetV3-Large": ROOT / "checkpoints" / "MobileNetV3-Large" / "best_model.pth",
}

SAVE_DIR = ROOT / "outputs" / "benchmarks" / "cnn_gpu"
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


NUM_CLASSES = 10

IMAGE_SIZE = 224

BATCH_SIZE = 1

# Pure forward benchmark
WARMUP_ITERS = 200
TIMING_ITERS = 1000

# Dataset evaluation
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Test transform
# ============================================================

# Unified transform for fair efficiency comparison.
#
# For pure model-forward timing, preprocessing does not affect
# the reported network latency.
#
# This transform is also used for end-to-end timing and
# checkpoint accuracy verification.

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Build models
# ============================================================

def build_model(model_name):

    if model_name == "AlexNet":

        model = models.alexnet(
            weights=None
        )

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            NUM_CLASSES,
        )

    elif model_name == "VGG-16":

        model = models.vgg16(
            weights=None
        )

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            NUM_CLASSES,
        )

    elif model_name == "EfficientNet-B0":

        model = models.efficientnet_b0(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES,
        )

    elif model_name == "MobileNetV3-Large":

        model = models.mobilenet_v3_large(
            weights=None
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            NUM_CLASSES,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# Flexible checkpoint loader
# ============================================================

def clean_state_dict(state_dict):

    new_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        if new_key.startswith("model."):
            new_key = new_key[6:]

        new_state_dict[new_key] = value

    return new_state_dict


def extract_state_dict(checkpoint):

    # --------------------------------------------------------
    # Case 1: checkpoint itself is a complete nn.Module
    # --------------------------------------------------------

    if isinstance(checkpoint, nn.Module):
        return checkpoint

    # --------------------------------------------------------
    # Case 2: direct state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        # A raw state_dict usually has tensor values
        tensor_values = [
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ]

        if (
            len(tensor_values) > 0
            and all(tensor_values)
        ):
            return checkpoint

        # Common checkpoint dictionary keys
        possible_keys = [
            "model_state_dict",
            "state_dict",
            "model",
            "model_state",
            "net",
            "network",
            "weights",
        ]

        for key in possible_keys:

            if key not in checkpoint:
                continue

            value = checkpoint[key]

            if isinstance(value, nn.Module):
                return value

            if isinstance(value, dict):
                return value

    raise RuntimeError(
        "Could not find model/state_dict in checkpoint."
    )


def load_model(model_name, checkpoint_path):

    print(
        f"\nLoading checkpoint:\n"
        f"{checkpoint_path}"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    # weights_only=False increases compatibility with
    # checkpoints saved using torch.save(model)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    extracted = extract_state_dict(
        checkpoint
    )

    # If a complete model was stored
    if isinstance(extracted, nn.Module):

        model = extracted

        print(
            "Checkpoint type: complete nn.Module"
        )

    else:

        model = build_model(
            model_name
        )

        state_dict = clean_state_dict(
            extracted
        )

        try:

            model.load_state_dict(
                state_dict,
                strict=True,
            )

        except RuntimeError as e:

            print("\n[ERROR] Strict model loading failed.")
            print(
                "This usually means the architecture used "
                "during training differs from torchvision."
            )

            print("\nCheckpoint first 20 keys:")

            for key in list(
                state_dict.keys()
            )[:20]:
                print("   ", key)

            print("\nModel first 20 keys:")

            for key in list(
                model.state_dict().keys()
            )[:20]:
                print("   ", key)

            raise e

        print(
            "Checkpoint type: state_dict"
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# Parameters / file size
# ============================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def file_size_mb(path):

    return (
        os.path.getsize(path)
        / 1024**2
    )


# ============================================================
# Pure forward inference benchmark
# ============================================================

@torch.no_grad()
def benchmark_forward(
    model,
    model_name,
):

    print("\n" + "-" * 70)
    print(
        f"{model_name}: pure GPU forward benchmark"
    )
    print("-" * 70)

    dummy = torch.randn(
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"Warm-up iterations: "
        f"{WARMUP_ITERS}"
    )

    for _ in range(
        WARMUP_ITERS
    ):
        _ = model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Reset memory statistics
    # --------------------------------------------------------

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(
            DEVICE
        )

        # current memory after model/input loaded
        baseline_gpu_mem = (
            torch.cuda.memory_allocated(
                DEVICE
            )
            / 1024**2
        )

    else:

        baseline_gpu_mem = 0.0

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    times_ms = []

    print(
        f"Timing iterations: "
        f"{TIMING_ITERS}"
    )

    for _ in range(
        TIMING_ITERS
    ):

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        _ = model(dummy)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - t0
        )

        times_ms.append(
            elapsed * 1000.0
        )

    times_ms = np.asarray(
        times_ms,
        dtype=float,
    )

    mean_ms = float(
        np.mean(times_ms)
    )

    std_ms = float(
        np.std(
            times_ms,
            ddof=1
        )
    )

    median_ms = float(
        np.median(times_ms)
    )

    p95_ms = float(
        np.percentile(
            times_ms,
            95
        )
    )

    fps = (
        BATCH_SIZE
        * 1000.0
        / mean_ms
    )

    if DEVICE.type == "cuda":

        peak_gpu_mem = (
            torch.cuda.max_memory_allocated(
                DEVICE
            )
            / 1024**2
        )

        peak_extra_mem = (
            peak_gpu_mem
            - baseline_gpu_mem
        )

    else:

        peak_gpu_mem = 0.0
        peak_extra_mem = 0.0

    print(
        f"Mean latency       : "
        f"{mean_ms:.4f} ms/image"
    )

    print(
        f"SD latency         : "
        f"{std_ms:.4f} ms"
    )

    print(
        f"Median latency     : "
        f"{median_ms:.4f} ms"
    )

    print(
        f"P95 latency        : "
        f"{p95_ms:.4f} ms"
    )

    print(
        f"FPS                : "
        f"{fps:.2f}"
    )

    if DEVICE.type == "cuda":

        print(
            f"Baseline GPU memory: "
            f"{baseline_gpu_mem:.2f} MB"
        )

        print(
            f"Peak GPU memory    : "
            f"{peak_gpu_mem:.2f} MB"
        )

        print(
            f"Peak extra memory  : "
            f"{peak_extra_mem:.2f} MB"
        )

    return {
        "Inference_mean_ms":
            mean_ms,

        "Inference_std_ms":
            std_ms,

        "Inference_median_ms":
            median_ms,

        "Inference_P95_ms":
            p95_ms,

        "Inference_FPS":
            fps,

        "GPU_baseline_MB":
            baseline_gpu_mem,

        "GPU_peak_MB":
            peak_gpu_mem,

        "GPU_peak_extra_MB":
            peak_extra_mem,
    }


# ============================================================
# Test-set accuracy
# ============================================================

@torch.no_grad()
def evaluate_accuracy(
    model,
    loader,
    model_name,
):

    print("\n" + "-" * 70)
    print(
        f"{model_name}: test accuracy verification"
    )
    print("-" * 70)

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=False,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=False,
        )

        outputs = model(
            images
        )

        pred = outputs.argmax(
            dim=1
        )

        correct += int(
            (pred == labels)
            .sum()
            .item()
        )

        total += int(
            labels.numel()
        )

    accuracy = (
        correct
        / total
        * 100.0
    )

    print(
        f"Correct : {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.4f}%"
    )

    return accuracy


# ============================================================
# End-to-end test-set benchmark
# ============================================================

@torch.no_grad()
def benchmark_end_to_end(
    model,
    dataset,
    model_name,
):

    """
    Includes:
    - image file reading
    - PIL decoding
    - Resize
    - ToTensor
    - normalization
    - host-to-device transfer
    - network forward

    batch size = 1
    num_workers = 0
    """

    print("\n" + "-" * 70)
    print(
        f"{model_name}: end-to-end benchmark"
    )
    print("-" * 70)

    # Important:
    # New DataLoader so timing begins from normal image loading.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    gc.collect()

    times_ms = []

    correct = 0
    total = 0

    overall_start = (
        time.perf_counter()
    )

    # Timing each complete sample cycle
    iterator = iter(loader)

    for _ in range(
        len(dataset)
    ):

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        images, labels = next(
            iterator
        )

        images = images.to(
            DEVICE,
            non_blocking=False,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=False,
        )

        outputs = model(
            images
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - t0
        )

        times_ms.append(
            elapsed * 1000.0
        )

        pred = outputs.argmax(
            dim=1
        )

        correct += int(
            (pred == labels)
            .sum()
            .item()
        )

        total += 1

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )

    times_ms = np.asarray(
        times_ms
    )

    mean_ms = float(
        np.mean(times_ms)
    )

    std_ms = float(
        np.std(
            times_ms,
            ddof=1
        )
    )

    median_ms = float(
        np.median(times_ms)
    )

    fps = (
        1000.0
        / mean_ms
    )

    whole_fps = (
        len(dataset)
        / overall_elapsed
    )

    accuracy = (
        correct
        / total
        * 100.0
    )

    print(
        f"Images             : "
        f"{len(dataset)}"
    )

    print(
        f"Mean end-to-end    : "
        f"{mean_ms:.4f} ms/image"
    )

    print(
        f"SD                 : "
        f"{std_ms:.4f} ms"
    )

    print(
        f"Median             : "
        f"{median_ms:.4f} ms"
    )

    print(
        f"End-to-end FPS     : "
        f"{fps:.2f}"
    )

    print(
        f"Whole-loop FPS     : "
        f"{whole_fps:.2f}"
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}%"
    )

    return {
        "End_to_end_mean_ms":
            mean_ms,

        "End_to_end_std_ms":
            std_ms,

        "End_to_end_median_ms":
            median_ms,

        "End_to_end_FPS":
            fps,

        "Whole_loop_FPS":
            whole_fps,

        "Accuracy_percent":
            accuracy,
    }


# ============================================================
# Main benchmark
# ============================================================

def main():

    print("=" * 80)
    print(
        "Reviewer 2 - Comment 7"
    )
    print(
        "FOUR DEEP-LEARNING MODELS"
        " EFFICIENCY BENCHMARK"
    )
    print("=" * 80)

    print(
        f"PyTorch version : "
        f"{torch.__version__}"
    )

    print(
        f"Device          : "
        f"{DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(DEVICE)}"
        )

        props = (
            torch.cuda.get_device_properties(
                DEVICE
            )
        )

        print(
            f"GPU total memory: "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

        print(
            f"CUDA version    : "
            f"{torch.version.cuda}"
        )

    else:

        raise RuntimeError(
            "CUDA GPU is required for this benchmark."
        )

    print(
        f"Input size      : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Batch size      : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Precision       : FP32"
    )

    print(
        f"Warm-up         : "
        f"{WARMUP_ITERS}"
    )

    print(
        f"Timed forwards  : "
        f"{TIMING_ITERS}"
    )

    print(
        f"Test directory  : "
        f"{TEST_DIR}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform,
    )

    print(
        f"\nTest images     : "
        f"{len(dataset)}"
    )

    print(
        f"Classes         : "
        f"{len(dataset.classes)}"
    )

    print("\nClass mapping:")

    for cls_name, cls_id in (
        dataset.class_to_idx.items()
    ):
        print(
            f"  {cls_id}: "
            f"{cls_name}"
        )

    accuracy_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    rows = []

    # --------------------------------------------------------
    # Four models
    # --------------------------------------------------------

    for model_name, ckpt_path in (
        MODEL_PATHS.items()
    ):

        print(
            "\n\n"
            + "=" * 80
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            "=" * 80
        )

        # Clean GPU before loading next model
        gc.collect()

        torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = load_model(
            model_name,
            ckpt_path,
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        total_params, trainable_params = (
            count_parameters(
                model
            )
        )

        params_m = (
            total_params
            / 1e6
        )

        trainable_m = (
            trainable_params
            / 1e6
        )

        ckpt_mb = file_size_mb(
            ckpt_path
        )

        print(
            f"\nParameters       : "
            f"{params_m:.3f} M"
        )

        print(
            f"Trainable params : "
            f"{trainable_m:.3f} M"
        )

        print(
            f"Checkpoint size  : "
            f"{ckpt_mb:.3f} MB"
        )

        # ----------------------------------------------------
        # Pure inference
        # ----------------------------------------------------

        forward_result = (
            benchmark_forward(
                model,
                model_name,
            )
        )

        # ----------------------------------------------------
        # Accuracy verification
        # ----------------------------------------------------

        accuracy = (
            evaluate_accuracy(
                model,
                accuracy_loader,
                model_name,
            )
        )

        # ----------------------------------------------------
        # End-to-end
        # ----------------------------------------------------

        e2e_result = (
            benchmark_end_to_end(
                model,
                dataset,
                model_name,
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        row = {
            "Method":
                model_name,

            "Input_size":
                f"{IMAGE_SIZE}x{IMAGE_SIZE}",

            "Batch_size":
                BATCH_SIZE,

            "Precision":
                "FP32",

            "Parameters_M":
                params_m,

            "Checkpoint_size_MB":
                ckpt_mb,

            "Inference_ms_per_image":
                forward_result[
                    "Inference_mean_ms"
                ],

            "Inference_SD_ms":
                forward_result[
                    "Inference_std_ms"
                ],

            "Inference_median_ms":
                forward_result[
                    "Inference_median_ms"
                ],

            "Inference_P95_ms":
                forward_result[
                    "Inference_P95_ms"
                ],

            "Inference_FPS":
                forward_result[
                    "Inference_FPS"
                ],

            "Peak_GPU_memory_MB":
                forward_result[
                    "GPU_peak_MB"
                ],

            "Extra_GPU_memory_MB":
                forward_result[
                    "GPU_peak_extra_MB"
                ],

            "End_to_end_ms_per_image":
                e2e_result[
                    "End_to_end_mean_ms"
                ],

            "End_to_end_FPS":
                e2e_result[
                    "End_to_end_FPS"
                ],

            "Accuracy_percent":
                accuracy,
        }

        rows.append(
            row
        )

        # ----------------------------------------------------
        # Free GPU
        # ----------------------------------------------------

        del model

        gc.collect()

        torch.cuda.empty_cache()

    # ========================================================
    # Final table
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    csv_path = (
        SAVE_DIR
        / "DL_4Models_efficiency_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n\n"
        + "=" * 100
    )

    print(
        "FINAL DEEP-LEARNING EFFICIENCY TABLE"
    )

    print(
        "=" * 100
    )

    display_cols = [
        "Method",
        "Parameters_M",
        "Checkpoint_size_MB",
        "Inference_ms_per_image",
        "Inference_FPS",
        "Peak_GPU_memory_MB",
        "End_to_end_ms_per_image",
        "End_to_end_FPS",
        "Accuracy_percent",
    ]

    print(
        df[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        csv_path
    )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()