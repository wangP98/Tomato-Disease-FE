# -*- coding: utf-8 -*-

"""
Reviewer 2 - Comment 7
CPU computational-efficiency benchmark for four CNN classifiers.

Models
------
1. AlexNet
2. VGG-16
3. EfficientNet-B0
4. MobileNetV3-Large

Metrics
-------
1. Parameter count
2. Approximate GMACs / GFLOPs (Conv2d + Linear)
3. Serialized model-state size
4. Pure CPU inference latency (ms/image)
5. Pure CPU inference FPS
6. End-to-end latency
   - image loading
   - image decoding
   - resize
   - tensor conversion
   - normalization
   - CNN forward inference
7. Peak CPU RAM
8. Optional test accuracy verification

Benchmark settings
------------------
- CPU only
- batch size = 1
- FP32
- input size = 224 x 224
- warm-up = 20 iterations
- timed forward = 200 iterations
- each model runs in an independent subprocess
- no psutil required
"""

# ============================================================
# IMPORTANT:
# These environment variables should be set before importing
# NumPy / PyTorch.
# ============================================================

import os

CPU_THREADS = 6

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)


# ============================================================
# Imports
# ============================================================

import sys
import gc
import json
import time
import platform
import argparse
import subprocess
import threading
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from PIL import Image

from torchvision import (
    datasets,
    transforms,
    models,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    ROOT
    / "data01"
    / "test"
)


MODEL_PATHS = {
    "AlexNet": ROOT / "checkpoints" / "AlexNet" / "best_model.pth",
    "VGG-16": ROOT / "checkpoints" / "VGG16" / "best_model.pth",
    "EfficientNet-B0": ROOT / "checkpoints" / "EfficientNet-B0" / "best_model.pth",
    "MobileNetV3-Large": ROOT / "checkpoints" / "MobileNetV3-Large" / "best_model.pth",
}

SAVE_DIR = ROOT / "outputs" / "benchmarks" / "cnn_cpu"
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Benchmark configuration
# ============================================================

DEVICE = torch.device("cpu")

NUM_CLASSES = 10

IMAGE_SIZE = 224

BATCH_SIZE = 1

DTYPE = torch.float32


# CPU warm-up
WARMUP_ITERS = 20


# Pure network forward timing
TIMING_ITERS = 200


# Number of real test images used for end-to-end timing.
# 200 is generally sufficient.
E2E_IMAGES = 200


# Optional accuracy verification.
#
# False:
#   fastest; only benchmark efficiency.
#
# True:
#   run the complete 3198-image test set for accuracy.
#
# The accuracy values are already available from previous
# experiments, so False is sufficient for Comment 7.
VERIFY_ACCURACY = False


# Memory sampling interval
MEM_INTERVAL = 0.002


# ============================================================
# PyTorch CPU configuration
# ============================================================

def configure_cpu():

    torch.set_num_threads(
        CPU_THREADS
    )

    try:
        torch.set_num_interop_threads(
            1
        )
    except RuntimeError:
        pass


# ============================================================
# Cross-platform process memory
# ============================================================

def get_current_rss_mb():
    """
    Current resident physical memory of this Python process.

    Linux:
        /proc/self/status

    Windows:
        GetProcessMemoryInfo()

    Returns
    -------
    float
        Current RSS in MB.
    """

    # --------------------------------------------------------
    # Linux
    # --------------------------------------------------------

    if os.name == "posix":

        try:

            with open(
                "/proc/self/status",
                "r"
            ) as f:

                for line in f:

                    if line.startswith(
                        "VmRSS:"
                    ):

                        value_kb = float(
                            line.split()[1]
                        )

                        return (
                            value_kb
                            / 1024.0
                        )

        except Exception:
            pass


    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    if os.name == "nt":

        try:

            import ctypes
            from ctypes import wintypes


            class PROCESS_MEMORY_COUNTERS(
                ctypes.Structure
            ):

                _fields_ = [

                    (
                        "cb",
                        wintypes.DWORD
                    ),

                    (
                        "PageFaultCount",
                        wintypes.DWORD
                    ),

                    (
                        "PeakWorkingSetSize",
                        ctypes.c_size_t
                    ),

                    (
                        "WorkingSetSize",
                        ctypes.c_size_t
                    ),

                    (
                        "QuotaPeakPagedPoolUsage",
                        ctypes.c_size_t
                    ),

                    (
                        "QuotaPagedPoolUsage",
                        ctypes.c_size_t
                    ),

                    (
                        "QuotaPeakNonPagedPoolUsage",
                        ctypes.c_size_t
                    ),

                    (
                        "QuotaNonPagedPoolUsage",
                        ctypes.c_size_t
                    ),

                    (
                        "PagefileUsage",
                        ctypes.c_size_t
                    ),

                    (
                        "PeakPagefileUsage",
                        ctypes.c_size_t
                    ),
                ]


            counters = (
                PROCESS_MEMORY_COUNTERS()
            )

            counters.cb = ctypes.sizeof(
                PROCESS_MEMORY_COUNTERS
            )


            process = (
                ctypes.windll.kernel32
                .GetCurrentProcess()
            )


            success = (
                ctypes.windll.psapi
                .GetProcessMemoryInfo(
                    process,
                    ctypes.byref(
                        counters
                    ),
                    counters.cb
                )
            )


            if success:

                return (
                    counters.WorkingSetSize
                    / 1024**2
                )

        except Exception:
            pass


    return 0.0


# ============================================================
# Peak RAM monitor
# ============================================================

class PeakMemoryMonitor:

    def __init__(
        self,
        interval=0.002
    ):

        self.interval = interval

        self.running = False

        self.thread = None

        self.start_rss = 0.0

        self.peak_rss = 0.0


    def _run(self):

        while self.running:

            rss = get_current_rss_mb()

            if rss > self.peak_rss:

                self.peak_rss = rss

            time.sleep(
                self.interval
            )


    def start(self):

        gc.collect()

        self.start_rss = (
            get_current_rss_mb()
        )

        self.peak_rss = (
            self.start_rss
        )

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()


    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join()


        end_rss = (
            get_current_rss_mb()
        )


        self.peak_rss = max(
            self.peak_rss,
            end_rss
        )


        return {

            "start_rss_mb":
                self.start_rss,

            "end_rss_mb":
                end_rss,

            "peak_rss_mb":
                self.peak_rss,

            "increment_mb":
                max(
                    0.0,
                    self.peak_rss
                    - self.start_rss
                ),
        }


# ============================================================
# Test preprocessing
# ============================================================

TEST_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# ============================================================
# Clean checkpoint keys
# ============================================================

def clean_state_dict(
    state_dict
):

    cleaned = {}


    prefixes = [

        "module.",

        "model.",

        "net.",

        "network.",
    ]


    for key, value in (
        state_dict.items()
    ):

        new_key = key


        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(
                    prefix
                ):

                    new_key = (
                        new_key[
                            len(prefix):
                        ]
                    )

                    changed = True


        cleaned[
            new_key
        ] = value


    return cleaned


# ============================================================
# Extract model/state_dict from checkpoint
# ============================================================

def extract_checkpoint_object(
    checkpoint
):

    # Complete PyTorch model
    if isinstance(
        checkpoint,
        nn.Module
    ):

        return checkpoint


    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Unsupported checkpoint type: "
            f"{type(checkpoint)}"
        )


    # --------------------------------------------------------
    # Is checkpoint itself a raw state_dict?
    # --------------------------------------------------------

    if len(checkpoint) > 0:

        tensor_values = [

            isinstance(
                value,
                torch.Tensor
            )

            for value
            in checkpoint.values()
        ]


        if all(
            tensor_values
        ):

            return checkpoint


    # --------------------------------------------------------
    # Common nested state_dict keys
    # --------------------------------------------------------

    candidate_keys = [

        "model_state_dict",

        "state_dict",

        "model_state",

        "model",

        "net",

        "network",

        "weights",

        "best_model",

        "best_state_dict",
    ]


    for key in candidate_keys:

        if key not in checkpoint:

            continue


        value = checkpoint[
            key
        ]


        if isinstance(
            value,
            nn.Module
        ):

            return value


        if isinstance(
            value,
            dict
        ):

            return value


    print(
        "\nAvailable checkpoint keys:"
    )

    for key in checkpoint.keys():

        print(
            f"  {key}"
        )


    raise RuntimeError(
        "Could not locate model state_dict."
    )


# ============================================================
# Build torchvision model
# ============================================================

def build_model(
    model_name,
    state_dict=None
):

    # --------------------------------------------------------
    # AlexNet
    # --------------------------------------------------------

    if model_name == "AlexNet":

        model = models.alexnet(
            weights=None
        )

        model.classifier[6] = (
            nn.Linear(
                model.classifier[
                    6
                ].in_features,
                NUM_CLASSES
            )
        )


    # --------------------------------------------------------
    # VGG16
    # Automatically detect VGG16 vs VGG16-BN if possible
    # --------------------------------------------------------

    elif model_name == "VGG-16":

        use_bn = False


        if state_dict is not None:

            # VGG16-BN contains BatchNorm parameters
            for key in (
                state_dict.keys()
            ):

                if (
                    "running_mean"
                    in key
                    and
                    key.startswith(
                        "features."
                    )
                ):

                    use_bn = True
                    break


        if use_bn:

            print(
                "Detected architecture: "
                "VGG16-BN"
            )

            model = (
                models.vgg16_bn(
                    weights=None
                )
            )

        else:

            print(
                "Detected architecture: "
                "VGG16"
            )

            model = (
                models.vgg16(
                    weights=None
                )
            )


        model.classifier[6] = (
            nn.Linear(
                model.classifier[
                    6
                ].in_features,
                NUM_CLASSES
            )
        )


    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    elif model_name == "EfficientNet-B0":

        model = (
            models.efficientnet_b0(
                weights=None
            )
        )

        model.classifier[1] = (
            nn.Linear(
                model.classifier[
                    1
                ].in_features,
                NUM_CLASSES
            )
        )


    # --------------------------------------------------------
    # MobileNetV3-Large
    # --------------------------------------------------------

    elif model_name == "MobileNetV3-Large":

        model = (
            models.mobilenet_v3_large(
                weights=None
            )
        )

        model.classifier[3] = (
            nn.Linear(
                model.classifier[
                    3
                ].in_features,
                NUM_CLASSES
            )
        )


    else:

        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )


    return model


# ============================================================
# Load checkpoint
# ============================================================

def load_model(
    model_name,
    checkpoint_path
):

    print(
        "\nCheckpoint:"
    )

    print(
        checkpoint_path
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )


    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )


    extracted = (
        extract_checkpoint_object(
            checkpoint
        )
    )


    # Complete nn.Module checkpoint
    if isinstance(
        extracted,
        nn.Module
    ):

        print(
            "Checkpoint format: "
            "complete nn.Module"
        )

        model = extracted


    else:

        print(
            "Checkpoint format: "
            "state_dict"
        )


        state_dict = (
            clean_state_dict(
                extracted
            )
        )


        model = build_model(
            model_name,
            state_dict
        )


        try:

            model.load_state_dict(
                state_dict,
                strict=True
            )

        except RuntimeError as exc:

            print(
                "\n"
                + "=" * 70
            )

            print(
                "STRICT MODEL LOADING FAILED"
            )

            print(
                "=" * 70
            )


            print(
                "\nFirst checkpoint keys:"
            )

            for key in list(
                state_dict.keys()
            )[:30]:

                value = (
                    state_dict[key]
                )

                shape = (
                    tuple(value.shape)
                    if hasattr(
                        value,
                        "shape"
                    )
                    else "?"
                )

                print(
                    f"{key:<50}"
                    f"{shape}"
                )


            print(
                "\nFirst model keys:"
            )

            model_sd = (
                model.state_dict()
            )

            for key in list(
                model_sd.keys()
            )[:30]:

                value = (
                    model_sd[key]
                )

                print(
                    f"{key:<50}"
                    f"{tuple(value.shape)}"
                )


            raise exc


    model = model.to(
        DEVICE
    )

    model.eval()


    return model


# ============================================================
# Parameter count
# ============================================================

def count_parameters(
    model
):

    total = sum(

        parameter.numel()

        for parameter
        in model.parameters()
    )


    trainable = sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


    return (
        total,
        trainable
    )


# ============================================================
# Model-state storage size
# ============================================================

def calculate_model_state_size_mb(
    model,
    model_name
):

    safe_name = (
        model_name
        .replace("-", "_")
        .replace(" ", "_")
    )


    temp_path = (
        SAVE_DIR
        / f"_tmp_{safe_name}_state.pth"
    )


    torch.save(
        model.state_dict(),
        temp_path
    )


    size_mb = (
        os.path.getsize(
            temp_path
        )
        / 1024**2
    )


    try:

        temp_path.unlink()

    except Exception:

        pass


    return size_mb


# ============================================================
# Approximate MACs/FLOPs
# Conv2d + Linear
# ============================================================

def calculate_macs_flops(
    model
):

    """
    Approximate computational complexity for batch size 1.

    Counts:
        Conv2d
        Linear

    Definitions:
        1 MAC ~= one multiply-accumulate
        Approximate FLOPs = 2 * MACs

    BN, activation, pooling and element-wise operations
    are not included, so this is an approximate value.
    """

    total_macs = 0

    hooks = []


    def conv_hook(
        module,
        inputs,
        output
    ):

        nonlocal total_macs


        # output:
        # [N, Cout, Hout, Wout]

        batch = output.shape[0]

        out_channels = (
            output.shape[1]
        )

        out_h = (
            output.shape[2]
        )

        out_w = (
            output.shape[3]
        )


        kernel_h = (
            module.kernel_size[0]
        )

        kernel_w = (
            module.kernel_size[1]
        )


        in_per_group = (
            module.in_channels
            // module.groups
        )


        macs = (

            batch
            * out_channels
            * out_h
            * out_w
            * in_per_group
            * kernel_h
            * kernel_w
        )


        total_macs += macs


    def linear_hook(
        module,
        inputs,
        output
    ):

        nonlocal total_macs


        input_tensor = (
            inputs[0]
        )


        batch = (
            input_tensor.numel()
            // module.in_features
        )


        macs = (

            batch
            * module.in_features
            * module.out_features
        )


        total_macs += macs


    for module in (
        model.modules()
    ):

        if isinstance(
            module,
            nn.Conv2d
        ):

            hooks.append(
                module.register_forward_hook(
                    conv_hook
                )
            )


        elif isinstance(
            module,
            nn.Linear
        ):

            hooks.append(
                module.register_forward_hook(
                    linear_hook
                )
            )


    dummy = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=DTYPE,
        device=DEVICE
    )


    with torch.inference_mode():

        _ = model(
            dummy
        )


    for hook in hooks:

        hook.remove()


    gmacs = (
        total_macs
        / 1e9
    )


    gflops = (
        2.0
        * total_macs
        / 1e9
    )


    return (
        gmacs,
        gflops
    )


# ============================================================
# Pure CPU model-forward benchmark
# ============================================================

def benchmark_forward(
    model
):

    print(
        "\n"
        + "-" * 70
    )

    print(
        "PURE CPU FORWARD INFERENCE"
    )

    print(
        "-" * 70
    )


    dummy = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=DTYPE,
        device=DEVICE
    )


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"Warm-up iterations : "
        f"{WARMUP_ITERS}"
    )


    with torch.inference_mode():

        for _ in range(
            WARMUP_ITERS
        ):

            _ = model(
                dummy
            )


    gc.collect()


    monitor = (
        PeakMemoryMonitor(
            MEM_INTERVAL
        )
    )

    monitor.start()


    times_ms = []


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    with torch.inference_mode():

        for _ in range(
            TIMING_ITERS
        ):

            t0 = (
                time.perf_counter()
            )


            _ = model(
                dummy
            )


            elapsed = (
                time.perf_counter()
                - t0
            )


            times_ms.append(
                elapsed * 1000.0
            )


    memory = (
        monitor.stop()
    )


    times_ms = np.asarray(
        times_ms,
        dtype=float
    )


    mean_ms = float(
        times_ms.mean()
    )


    std_ms = float(
        times_ms.std(
            ddof=1
        )
    )


    median_ms = float(
        np.median(
            times_ms
        )
    )


    p95_ms = float(
        np.percentile(
            times_ms,
            95
        )
    )


    fps = (
        1000.0
        / mean_ms
    )


    print(
        f"Timed iterations   : "
        f"{TIMING_ITERS}"
    )

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Mean latency       : "
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
        f"P95                : "
        f"{p95_ms:.4f} ms"
    )

    print(
        f"CPU inference FPS  : "
        f"{fps:.2f}"
    )

    print(
        f"Peak CPU RAM       : "
        f"{memory['peak_rss_mb']:.2f} MB"
    )

    print(
        f"RAM increment      : "
        f"{memory['increment_mb']:.2f} MB"
    )


    return {

        "mean_ms":
            mean_ms,

        "std_ms":
            std_ms,

        "median_ms":
            median_ms,

        "p95_ms":
            p95_ms,

        "fps":
            fps,

        "peak_ram_mb":
            memory[
                "peak_rss_mb"
            ],

        "ram_increment_mb":
            memory[
                "increment_mb"
            ],
    }


# ============================================================
# End-to-end benchmark
# ============================================================

def benchmark_end_to_end(
    model,
    dataset
):

    print(
        "\n"
        + "-" * 70
    )

    print(
        "END-TO-END CPU BENCHMARK"
    )

    print(
        "-" * 70
    )


    n_images = min(
        E2E_IMAGES,
        len(dataset.samples)
    )


    if n_images <= 0:

        raise RuntimeError(
            "No test images found."
        )


    # Warm up network separately
    dummy = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )


    with torch.inference_mode():

        for _ in range(5):

            _ = model(
                dummy
            )


    gc.collect()


    monitor = (
        PeakMemoryMonitor(
            MEM_INTERVAL
        )
    )

    monitor.start()


    times_ms = []


    with torch.inference_mode():

        for i in range(
            n_images
        ):

            image_path, _ = (
                dataset.samples[i]
            )


            t0 = (
                time.perf_counter()
            )


            # ================================================
            # Image reading + decoding
            # ================================================

            with Image.open(
                image_path
            ) as image:

                image = image.convert(
                    "RGB"
                )


                # ============================================
                # Resize + ToTensor + Normalize
                # ============================================

                image_tensor = (
                    TEST_TRANSFORM(
                        image
                    )
                )


            image_tensor = (
                image_tensor
                .unsqueeze(0)
            )


            # ================================================
            # CNN forward
            # ================================================

            _ = model(
                image_tensor
            )


            elapsed = (
                time.perf_counter()
                - t0
            )


            times_ms.append(
                elapsed * 1000.0
            )


    memory = (
        monitor.stop()
    )


    times_ms = np.asarray(
        times_ms,
        dtype=float
    )


    mean_ms = float(
        times_ms.mean()
    )


    std_ms = float(
        times_ms.std(
            ddof=1
        )
    )


    median_ms = float(
        np.median(
            times_ms
        )
    )


    p95_ms = float(
        np.percentile(
            times_ms,
            95
        )
    )


    fps = (
        1000.0
        / mean_ms
    )


    print(
        f"Images tested      : "
        f"{n_images}"
    )

    print(
        f"Mean latency       : "
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
        f"P95                : "
        f"{p95_ms:.4f} ms"
    )

    print(
        f"End-to-end FPS     : "
        f"{fps:.2f}"
    )

    print(
        f"Peak CPU RAM       : "
        f"{memory['peak_rss_mb']:.2f} MB"
    )


    return {

        "n_images":
            n_images,

        "mean_ms":
            mean_ms,

        "std_ms":
            std_ms,

        "median_ms":
            median_ms,

        "p95_ms":
            p95_ms,

        "fps":
            fps,

        "peak_ram_mb":
            memory[
                "peak_rss_mb"
            ],
    }


# ============================================================
# Optional full-test accuracy
# ============================================================

def verify_accuracy(
    model,
    dataset
):

    print(
        "\n"
        + "-" * 70
    )

    print(
        "FULL TEST ACCURACY"
    )

    print(
        "-" * 70
    )


    correct = 0

    total = 0


    with torch.inference_mode():

        for i in range(
            len(dataset.samples)
        ):

            image_path, label = (
                dataset.samples[i]
            )


            with Image.open(
                image_path
            ) as image:

                image = (
                    image.convert(
                        "RGB"
                    )
                )

                x = (
                    TEST_TRANSFORM(
                        image
                    )
                )


            x = x.unsqueeze(
                0
            )


            output = model(
                x
            )


            pred = int(
                output.argmax(
                    dim=1
                ).item()
            )


            correct += int(
                pred == label
            )

            total += 1


            if (
                (i + 1)
                % 500
                == 0
            ):

                print(
                    f"{i + 1}/"
                    f"{len(dataset)}"
                )


    accuracy = (
        correct
        / total
        * 100.0
    )


    print(
        f"Correct : "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}%"
    )


    return accuracy


# ============================================================
# Worker: benchmark one model
# ============================================================

def benchmark_one_model(
    model_name
):

    configure_cpu()


    checkpoint_path = (
        MODEL_PATHS[
            model_name
        ]
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"MODEL: {model_name}"
    )

    print(
        "=" * 80
    )


    print(
        f"Device            : CPU"
    )

    print(
        f"PyTorch           : "
        f"{torch.__version__}"
    )

    print(
        f"CPU threads       : "
        f"{CPU_THREADS}"
    )

    print(
        f"Input             : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size        : "
        f"{BATCH_SIZE}"
    )

    print(
        "Precision         : FP32"
    )


    # --------------------------------------------------------
    # Process RAM before model load
    # --------------------------------------------------------

    gc.collect()

    baseline_rss = (
        get_current_rss_mb()
    )


    load_monitor = (
        PeakMemoryMonitor(
            MEM_INTERVAL
        )
    )

    load_monitor.start()


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_name,
        checkpoint_path
    )


    load_memory = (
        load_monitor.stop()
    )


    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    (
        total_params,
        trainable_params,
    ) = count_parameters(
        model
    )


    params_m = (
        total_params
        / 1e6
    )


    # --------------------------------------------------------
    # Model state size
    # --------------------------------------------------------

    model_size_mb = (
        calculate_model_state_size_mb(
            model,
            model_name
        )
    )


    checkpoint_size_mb = (
        os.path.getsize(
            checkpoint_path
        )
        / 1024**2
    )


    # --------------------------------------------------------
    # Complexity
    # --------------------------------------------------------

    (
        gmacs,
        gflops,
    ) = calculate_macs_flops(
        model
    )


    print(
        "\nModel complexity:"
    )

    print(
        f"Parameters         : "
        f"{params_m:.3f} M"
    )

    print(
        f"GMACs              : "
        f"{gmacs:.3f}"
    )

    print(
        f"Approx. GFLOPs     : "
        f"{gflops:.3f}"
    )

    print(
        f"Model-state size   : "
        f"{model_size_mb:.3f} MB"
    )

    print(
        f"Checkpoint size    : "
        f"{checkpoint_size_mb:.3f} MB"
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = datasets.ImageFolder(
        TEST_DIR
    )


    print(
        f"\nTest images       : "
        f"{len(dataset)}"
    )

    print(
        f"Classes           : "
        f"{len(dataset.classes)}"
    )


    # --------------------------------------------------------
    # Pure CPU inference
    # --------------------------------------------------------

    forward_result = (
        benchmark_forward(
            model
        )
    )


    # --------------------------------------------------------
    # End-to-end CPU timing
    # --------------------------------------------------------

    e2e_result = (
        benchmark_end_to_end(
            model,
            dataset
        )
    )


    # --------------------------------------------------------
    # Optional accuracy verification
    # --------------------------------------------------------

    if VERIFY_ACCURACY:

        accuracy = (
            verify_accuracy(
                model,
                dataset
            )
        )

    else:

        accuracy = None


    # --------------------------------------------------------
    # Overall peak RAM
    # --------------------------------------------------------

    overall_peak_ram = max(

        load_memory[
            "peak_rss_mb"
        ],

        forward_result[
            "peak_ram_mb"
        ],

        e2e_result[
            "peak_ram_mb"
        ],
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "Method":
            model_name,

        "Device":
            "CPU",

        "CPU_threads":
            CPU_THREADS,

        "Input_size":
            f"{IMAGE_SIZE}x{IMAGE_SIZE}",

        "Batch_size":
            BATCH_SIZE,

        "Precision":
            "FP32",

        "Parameters_M":
            params_m,

        "GMACs":
            gmacs,

        "Approx_GFLOPs":
            gflops,

        "Model_state_size_MB":
            model_size_mb,

        "Checkpoint_size_MB":
            checkpoint_size_mb,

        "Inference_mean_ms":
            forward_result[
                "mean_ms"
            ],

        "Inference_SD_ms":
            forward_result[
                "std_ms"
            ],

        "Inference_median_ms":
            forward_result[
                "median_ms"
            ],

        "Inference_P95_ms":
            forward_result[
                "p95_ms"
            ],

        "Inference_FPS":
            forward_result[
                "fps"
            ],

        "Inference_peak_RAM_MB":
            forward_result[
                "peak_ram_mb"
            ],

        "End_to_end_images":
            e2e_result[
                "n_images"
            ],

        "End_to_end_mean_ms":
            e2e_result[
                "mean_ms"
            ],

        "End_to_end_SD_ms":
            e2e_result[
                "std_ms"
            ],

        "End_to_end_median_ms":
            e2e_result[
                "median_ms"
            ],

        "End_to_end_P95_ms":
            e2e_result[
                "p95_ms"
            ],

        "End_to_end_FPS":
            e2e_result[
                "fps"
            ],

        "End_to_end_peak_RAM_MB":
            e2e_result[
                "peak_ram_mb"
            ],

        "Overall_peak_RAM_MB":
            overall_peak_ram,

        "Process_baseline_RAM_MB":
            baseline_rss,

        "Model_load_increment_RAM_MB":
            max(
                0.0,
                load_memory[
                    "peak_rss_mb"
                ]
                - baseline_rss
            ),

        "Accuracy_percent":
            accuracy,
    }


    # --------------------------------------------------------
    # Save individual JSON
    # --------------------------------------------------------

    safe_name = (
        model_name
        .replace("-", "_")
        .replace(" ", "_")
    )


    json_path = (
        SAVE_DIR
        / f"{safe_name}_CPU_result.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"FINAL RESULT: "
        f"{model_name}"
    )

    print(
        "=" * 80
    )


    print(
        f"Parameters          : "
        f"{params_m:.3f} M"
    )

    print(
        f"Approx. GFLOPs      : "
        f"{gflops:.3f}"
    )

    print(
        f"Model size          : "
        f"{model_size_mb:.3f} MB"
    )

    print(
        f"CPU inference       : "
        f"{forward_result['mean_ms']:.4f} "
        f"ms/image"
    )

    print(
        f"CPU inference FPS   : "
        f"{forward_result['fps']:.2f}"
    )

    print(
        f"End-to-end latency  : "
        f"{e2e_result['mean_ms']:.4f} "
        f"ms/image"
    )

    print(
        f"End-to-end FPS      : "
        f"{e2e_result['fps']:.2f}"
    )

    print(
        f"Peak CPU RAM        : "
        f"{overall_peak_ram:.2f} MB"
    )


    if accuracy is not None:

        print(
            f"Accuracy            : "
            f"{accuracy:.4f}%"
        )


    print(
        f"\nSaved:"
    )

    print(
        json_path
    )


    return result


# ============================================================
# Parent process:
# run each model in a fresh Python process
# ============================================================

def run_all_models():

    print(
        "=" * 90
    )

    print(
        "Reviewer 2 - Comment 7"
    )

    print(
        "FOUR CNN MODELS - CPU EFFICIENCY BENCHMARK"
    )

    print(
        "=" * 90
    )


    print(
        f"\nCPU threads per model : "
        f"{CPU_THREADS}"
    )

    print(
        f"Device                : CPU"
    )

    print(
        f"Input                 : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size            : 1"
    )

    print(
        f"Precision             : FP32"
    )

    print(
        f"Warm-up               : "
        f"{WARMUP_ITERS}"
    )

    print(
        f"Forward timing        : "
        f"{TIMING_ITERS}"
    )

    print(
        f"End-to-end images     : "
        f"{E2E_IMAGES}"
    )


    print(
        "\nEach model is benchmarked in "
        "an independent Python process."
    )


    script_path = Path(
        __file__
    ).resolve()


    for model_name in (
        MODEL_PATHS.keys()
    ):

        print(
            "\n\n"
            + "#" * 90
        )

        print(
            f"STARTING: "
            f"{model_name}"
        )

        print(
            "#" * 90
        )


        command = [

            sys.executable,

            str(
                script_path
            ),

            "--worker",

            model_name,
        ]


        result = subprocess.run(
            command
        )


        if result.returncode != 0:

            raise RuntimeError(
                f"\nBenchmark failed for "
                f"{model_name}.\n"
                f"Return code: "
                f"{result.returncode}"
            )


    # --------------------------------------------------------
    # Read worker results
    # --------------------------------------------------------

    rows = []


    for model_name in (
        MODEL_PATHS.keys()
    ):

        safe_name = (
            model_name
            .replace("-", "_")
            .replace(" ", "_")
        )


        json_path = (
            SAVE_DIR
            / f"{safe_name}_CPU_result.json"
        )


        if not json_path.exists():

            raise FileNotFoundError(
                json_path
            )


        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            rows.append(
                json.load(
                    f
                )
            )


    df = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Save full CSV
    # --------------------------------------------------------

    csv_path = (
        SAVE_DIR
        / "DL_4Models_CPU_efficiency_results.csv"
    )


    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # Paper-ready simplified table
    # --------------------------------------------------------

    paper_cols = [

        "Method",

        "Device",

        "Parameters_M",

        "Approx_GFLOPs",

        "Model_state_size_MB",

        "Inference_mean_ms",

        "Inference_FPS",

        "End_to_end_mean_ms",

        "End_to_end_FPS",

        "Overall_peak_RAM_MB",
    ]


    paper_df = df[
        paper_cols
    ].copy()


    paper_path = (
        SAVE_DIR
        / "DL_4Models_CPU_paper_table.csv"
    )


    paper_df.to_csv(
        paper_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "\n\n"
        + "=" * 120
    )

    print(
        "FINAL CPU EFFICIENCY TABLE"
    )

    print(
        "=" * 120
    )


    print(

        paper_df.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.4f}"
        )
    )


    print(
        "\nFull results:"
    )

    print(
        csv_path
    )


    print(
        "\nPaper table:"
    )

    print(
        paper_path
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--worker",
        type=str,
        default=None,
        choices=list(
            MODEL_PATHS.keys()
        ),
        help=(
            "Internal option: "
            "benchmark one model."
        )
    )


    args = parser.parse_args()


    if args.worker is not None:

        benchmark_one_model(
            args.worker
        )

    else:

        run_all_models()


if __name__ == "__main__":

    main()