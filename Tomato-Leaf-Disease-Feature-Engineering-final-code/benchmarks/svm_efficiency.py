# -*- coding: utf-8 -*-

"""
Reviewer 2 - Comment 7
Computational efficiency benchmark for the proposed
handcrafted-feature + Pruned SVM pipeline.

Metrics
-------
1. Handcrafted feature extraction time
2. Feature preprocessing time
   - StandardScaler
   - pruned-feature selection
3. Pure Pruned SVM inference time
4. End-to-end latency per image
5. Throughput / FPS
6. Peak CPU RAM
7. Serialized Pruned SVM package size
8. Test Accuracy and Macro-F1 for verification

Important
---------
- NO training
- NO permutation importance
- NO feature selection
- NO psutil required
- Linux /proc is used for CPU RAM measurement
- SVM timing conditions:
    batch size = 1
    warm-up = 200
    timed iterations = 1000
"""

import os
import sys
import gc
import time
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


TEST_DIR = (
    ROOT
    / "data01"
    / "test"
)

DATA_ROOT = (
    ROOT
    / "data01"
)

MASK_ROOT = (
    ROOT
    / "masks"
)


MODEL_FILE = ROOT / "checkpoints" / "svm_full_and_pruned.joblib"

SAVE_DIR = ROOT / "outputs" / "benchmarks" / "svm_cpu"
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Project feature extractor
# ============================================================

from util.features_common import (
    load_split_features,
)


FEAT_KW = dict(
    use_color=True,
    use_lbp=True,
    use_shape=True,
    use_lesion=True,
)


# ============================================================
# Benchmark configuration
# ============================================================

# Keep these identical to the CNN benchmark
BATCH_SIZE = 1

WARMUP_ITERS = 200

TIMING_ITERS = 1000

# Interval for CPU RAM monitoring
MEM_INTERVAL = 0.005


# ============================================================
# CPU memory monitor
# Linux /proc implementation
# NO psutil
# ============================================================

def get_current_rss_mb():
    """
    Read current resident memory (RSS) of this
    Python process from /proc/self/status.

    Returns
    -------
    float
        Current RSS in MB.
    """

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

    return 0.0


class PeakMemoryMonitor:
    """
    Monitor peak RSS of the current Python process.

    No psutil or external package is required.
    """

    def __init__(
        self,
        interval=0.005
    ):

        self.interval = interval

        self.running = False

        self.thread = None

        self.start_rss = 0.0

        self.peak_rss = 0.0


    def _monitor(self):

        while self.running:

            rss = (
                get_current_rss_mb()
            )

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

        self.thread = (
            threading.Thread(
                target=self._monitor,
                daemon=True
            )
        )

        self.thread.start()


    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join()

        end_rss = (
            get_current_rss_mb()
        )

        if end_rss > self.peak_rss:

            self.peak_rss = end_rss

        return {
            "start_rss_mb":
                self.start_rss,

            "end_rss_mb":
                end_rss,

            "peak_rss_mb":
                self.peak_rss,

            "peak_increment_mb":
                max(
                    0.0,
                    self.peak_rss
                    - self.start_rss
                ),
        }


# ============================================================
# Load trained model
# ============================================================

def load_old_model():

    print("\n" + "=" * 75)
    print("LOAD SAVED SVM")
    print("=" * 75)

    print("Model file:")
    print(MODEL_FILE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    obj = joblib.load(MODEL_FILE)

    scaler = obj["scaler"]
    model_pruned = obj["model_pruned"]
    feature_names = obj["feature_names"]
    pruned_feature_names = obj["pruned_feature_names"]

    print(
        f"Full features   : "
        f"{len(feature_names)}"
    )
    print(
        f"Pruned features : "
        f"{len(pruned_feature_names)}"
    )

    return (
        scaler,
        model_pruned,
        feature_names,
        pruned_feature_names,
    )


# ============================================================
# Feature extraction benchmark
# ============================================================

def benchmark_feature_extraction():

    print(
        "\n"
        + "=" * 75
    )

    print(
        "1. HANDCRAFTED FEATURE EXTRACTION"
    )

    print(
        "=" * 75
    )


    print(
        "Test directory:"
    )

    print(
        TEST_DIR
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Features are re-extracted "
        "directly from test images."
    )

    print(
        "Cached X_test_raw.npy is NOT "
        "used for timing."
    )


    if not TEST_DIR.exists():

        raise FileNotFoundError(
            TEST_DIR
        )


    gc.collect()


    monitor = PeakMemoryMonitor(
        MEM_INTERVAL
    )

    monitor.start()


    t0 = time.perf_counter()


    X, y = load_split_features(
        TEST_DIR,
        mask_root=MASK_ROOT,
        common_root=DATA_ROOT,
        **FEAT_KW,
    )


    elapsed = (
        time.perf_counter()
        - t0
    )


    mem = monitor.stop()


    X = np.asarray(
        X
    )

    y = np.asarray(
        y
    ).reshape(-1)


    if X.ndim != 2:

        raise ValueError(
            f"Expected 2-D feature matrix, "
            f"got {X.shape}"
        )


    if len(X) != len(y):

        raise ValueError(
            f"X/y mismatch: "
            f"{len(X)} vs {len(y)}"
        )


    n = len(y)


    ms_per_image = (
        elapsed
        / n
        * 1000.0
    )


    fps = (
        n
        / elapsed
    )


    print(
        "\nFeature matrix:"
    )

    print(
        f"X: {X.shape}"
    )

    print(
        f"y: {y.shape}"
    )


    print(
        "\nFeature extraction:"
    )

    print(
        f"Total time        : "
        f"{elapsed:.4f} s"
    )

    print(
        f"Time / image      : "
        f"{ms_per_image:.4f} ms"
    )

    print(
        f"Throughput        : "
        f"{fps:.2f} images/s"
    )


    print(
        "\nCPU memory:"
    )

    print(
        f"Start RSS         : "
        f"{mem['start_rss_mb']:.2f} MB"
    )

    print(
        f"Peak RSS          : "
        f"{mem['peak_rss_mb']:.2f} MB"
    )

    print(
        f"Peak increment    : "
        f"{mem['peak_increment_mb']:.2f} MB"
    )


    result = {

        "total_time_s":
            elapsed,

        "ms_per_image":
            ms_per_image,

        "fps":
            fps,

        "start_rss_mb":
            mem["start_rss_mb"],

        "peak_rss_mb":
            mem["peak_rss_mb"],

        "peak_increment_mb":
            mem[
                "peak_increment_mb"
            ],
    }


    return (
        X,
        y,
        result
    )


# ============================================================
# Construct pruned feature indices
# ============================================================

def build_pruned_indices(
    feature_names,
    pruned_feature_names
):

    name_to_index = {
        name: i
        for i, name
        in enumerate(
            feature_names
        )
    }


    missing = [

        name

        for name
        in pruned_feature_names

        if name
        not in name_to_index
    ]


    if missing:

        print(
            "\nMissing feature names:"
        )

        for name in missing[:20]:

            print(
                name
            )

        raise ValueError(
            f"{len(missing)} pruned "
            f"features were not found."
        )


    indices = np.asarray(
        [
            name_to_index[name]
            for name
            in pruned_feature_names
        ],
        dtype=int
    )


    return indices


# ============================================================
# Full matrix preprocessing
# ============================================================

def preprocess_full_matrix(
    X,
    scaler,
    pruned_indices,
):

    X_scaled = scaler.transform(X)

    return X_scaled[:, pruned_indices]


# ============================================================
# Single-image preprocessing
# ============================================================

def preprocess_one_image(
    x,
    scaler,
    pruned_indices,
):

    x_scaled = scaler.transform(x)

    return x_scaled[:, pruned_indices]


# ============================================================
# Test accuracy verification
# ============================================================

def verify_model(
    X,
    y,
    scaler,
    model_pruned,
    pruned_indices,
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2. MODEL VERIFICATION"
    )

    print(
        "=" * 75
    )


    X_pruned = (
        preprocess_full_matrix(
            X,
            scaler,
            pruned_indices,
        )
    )


    pred = model_pruned.predict(
        X_pruned
    )


    acc = accuracy_score(
        y,
        pred
    )


    macro_f1 = f1_score(
        y,
        pred,
        average="macro",
        zero_division=0,
    )


    print(
        f"Samples            : "
        f"{len(y)}"
    )

    print(
        f"Pruned dimension   : "
        f"{X_pruned.shape[1]}"
    )

    print(
        f"Accuracy           : "
        f"{acc * 100:.2f}%"
    )

    print(
        f"Macro-F1           : "
        f"{macro_f1 * 100:.2f}%"
    )


    return (
        X_pruned,
        acc,
        macro_f1,
    )


# ============================================================
# Single-image preprocessing benchmark
# ============================================================

def benchmark_preprocessing(
    X,
    scaler,
    pruned_indices,
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        "3. SINGLE-IMAGE FEATURE PREPROCESSING"
    )

    print(
        "=" * 75
    )


    n = len(X)


    # -------------------------
    # Warm-up
    # -------------------------

    for i in range(
        WARMUP_ITERS
    ):

        idx = i % n

        x = X[
            idx:idx + 1
        ]

        _ = preprocess_one_image(
            x,
            scaler,
            pruned_indices,
        )


    # -------------------------
    # Timing
    # -------------------------

    times_ms = []


    for i in range(
        TIMING_ITERS
    ):

        idx = i % n

        x = X[
            idx:idx + 1
        ]


        t0 = (
            time.perf_counter()
        )


        _ = preprocess_one_image(
            x,
            scaler,
            pruned_indices,
        )


        elapsed = (
            time.perf_counter()
            - t0
        )


        times_ms.append(
            elapsed * 1000.0
        )


    times_ms = np.asarray(
        times_ms,
        dtype=float
    )


    mean_ms = float(
        np.mean(
            times_ms
        )
    )


    std_ms = float(
        np.std(
            times_ms,
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


    print(
        f"Warm-up iterations : "
        f"{WARMUP_ITERS}"
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
        f"{mean_ms:.6f} ms/image"
    )

    print(
        f"SD                 : "
        f"{std_ms:.6f} ms"
    )

    print(
        f"Median             : "
        f"{median_ms:.6f} ms"
    )

    print(
        f"P95                : "
        f"{p95_ms:.6f} ms"
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
    }


# ============================================================
# Single-image SVM inference benchmark
# ============================================================

def benchmark_svm_inference(
    X_pruned,
    model_pruned,
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        "4. PURE PRUNED SVM INFERENCE"
    )

    print(
        "=" * 75
    )


    n = len(
        X_pruned
    )


    # -------------------------
    # Warm-up
    # -------------------------

    for i in range(
        WARMUP_ITERS
    ):

        idx = i % n

        x = X_pruned[
            idx:idx + 1
        ]

        _ = model_pruned.predict(
            x
        )


    gc.collect()


    monitor = PeakMemoryMonitor(
        MEM_INTERVAL
    )

    monitor.start()


    # -------------------------
    # Timing
    # -------------------------

    times_ms = []


    for i in range(
        TIMING_ITERS
    ):

        idx = i % n

        x = X_pruned[
            idx:idx + 1
        ]


        t0 = (
            time.perf_counter()
        )


        _ = model_pruned.predict(
            x
        )


        elapsed = (
            time.perf_counter()
            - t0
        )


        times_ms.append(
            elapsed * 1000.0
        )


    mem = monitor.stop()


    times_ms = np.asarray(
        times_ms,
        dtype=float
    )


    mean_ms = float(
        np.mean(
            times_ms
        )
    )


    std_ms = float(
        np.std(
            times_ms,
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
        f"Warm-up iterations : "
        f"{WARMUP_ITERS}"
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
        f"{mean_ms:.6f} ms/image"
    )

    print(
        f"SD                 : "
        f"{std_ms:.6f} ms"
    )

    print(
        f"Median             : "
        f"{median_ms:.6f} ms"
    )

    print(
        f"P95                : "
        f"{p95_ms:.6f} ms"
    )

    print(
        f"FPS                : "
        f"{fps:.2f}"
    )


    print(
        "\nCPU memory during SVM inference:"
    )

    print(
        f"Start RSS          : "
        f"{mem['start_rss_mb']:.2f} MB"
    )

    print(
        f"Peak RSS           : "
        f"{mem['peak_rss_mb']:.2f} MB"
    )

    print(
        f"Peak increment     : "
        f"{mem['peak_increment_mb']:.2f} MB"
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

        "peak_rss_mb":
            mem["peak_rss_mb"],

        "peak_increment_mb":
            mem[
                "peak_increment_mb"
            ],
    }


# ============================================================
# Serialized SVM model size
# ============================================================

def benchmark_model_size(
    scaler,
    model_pruned,
    feature_names,
    pruned_feature_names,
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        "5. MODEL STORAGE SIZE"
    )

    print(
        "=" * 75
    )


    package = {

        "scaler":
            scaler,

        "model_pruned":
            model_pruned,

        "feature_names":
            feature_names,

        "pruned_feature_names":
            pruned_feature_names,
    }


    temp_path = (
        SAVE_DIR
        / "_temporary_pruned_svm.joblib"
    )


    joblib.dump(
        package,
        temp_path,
        compress=0,
    )


    size_bytes = os.path.getsize(
        temp_path
    )


    size_mb = (
        size_bytes
        / 1024**2
    )


    print(
        f"Pruned SVM package size : "
        f"{size_mb:.4f} MB"
    )


    try:

        temp_path.unlink()

    except Exception:

        pass


    return size_mb


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "Reviewer 2 - Comment 7"
    )

    print(
        "PROPOSED METHOD EFFICIENCY BENCHMARK"
    )

    print(
        "=" * 80
    )


    print(
        f"\nProject root      : "
        f"{ROOT}"
    )

    print(
        f"Batch size        : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Warm-up iterations: "
        f"{WARMUP_ITERS}"
    )

    print(
        f"Timed iterations  : "
        f"{TIMING_ITERS}"
    )


    # ========================================================
    # Load model first
    # ========================================================

    (
        scaler,
        model_pruned,
        feature_names,
        pruned_feature_names,
    ) = load_old_model()


    pruned_indices = (
        build_pruned_indices(
            feature_names,
            pruned_feature_names
        )
    )


    # ========================================================
    # Feature extraction
    # ========================================================

    (
        X,
        y,
        feat_result,
    ) = benchmark_feature_extraction()


    if X.shape[1] != len(
        feature_names
    ):

        raise ValueError(
            f"Feature dimension mismatch: "
            f"extracted X has {X.shape[1]}, "
            f"saved model expects "
            f"{len(feature_names)}."
        )


    # ========================================================
    # Verify accuracy
    # ========================================================

    (
        X_pruned,
        accuracy,
        macro_f1,
    ) = verify_model(
        X,
        y,
        scaler,
        model_pruned,
        pruned_indices,
    )


    # ========================================================
    # Preprocessing timing
    # ========================================================

    prep_result = (
        benchmark_preprocessing(
            X,
            scaler,
            pruned_indices,
        )
    )


    # ========================================================
    # Pure SVM timing
    # ========================================================

    svm_result = (
        benchmark_svm_inference(
            X_pruned,
            model_pruned,
        )
    )


    # ========================================================
    # Model storage size
    # ========================================================

    model_size_mb = (
        benchmark_model_size(
            scaler,
            model_pruned,
            feature_names,
            pruned_feature_names,
        )
    )


    # ========================================================
    # End-to-end
    # ========================================================

    feature_ms = (
        feat_result[
            "ms_per_image"
        ]
    )


    preprocessing_ms = (
        prep_result[
            "mean_ms"
        ]
    )


    svm_ms = (
        svm_result[
            "mean_ms"
        ]
    )


    total_ms = (
        feature_ms
        + preprocessing_ms
        + svm_ms
    )


    total_fps = (
        1000.0
        / total_ms
    )


    # ========================================================
    # Final output
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL PROPOSED PIPELINE RESULTS"
    )

    print(
        "=" * 80
    )


    print(
        f"Test images              : "
        f"{len(y)}"
    )

    print(
        f"Full feature dimension   : "
        f"{len(feature_names)}"
    )

    print(
        f"Pruned feature dimension : "
        f"{len(pruned_feature_names)}"
    )


    print(
        "\nPerformance verification:"
    )

    print(
        f"Accuracy                 : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1                 : "
        f"{macro_f1 * 100:.2f}%"
    )


    print(
        "\nEfficiency:"
    )

    print(
        f"Feature extraction       : "
        f"{feature_ms:.4f} ms/image"
    )

    print(
        f"Feature preprocessing    : "
        f"{preprocessing_ms:.4f} ms/image"
    )

    print(
        f"Pure SVM inference       : "
        f"{svm_ms:.4f} ms/image"
    )

    print(
        f"End-to-end latency       : "
        f"{total_ms:.4f} ms/image"
    )

    print(
        f"Pure SVM FPS             : "
        f"{svm_result['fps']:.2f}"
    )

    print(
        f"End-to-end throughput    : "
        f"{total_fps:.2f} images/s"
    )


    print(
        "\nMemory and storage:"
    )

    print(
        f"Feature extraction "
        f"peak RAM                 : "
        f"{feat_result['peak_rss_mb']:.2f} MB"
    )

    print(
        f"Feature extraction "
        f"additional RAM           : "
        f"{feat_result['peak_increment_mb']:.2f} MB"
    )

    print(
        f"SVM inference peak RAM   : "
        f"{svm_result['peak_rss_mb']:.2f} MB"
    )

    print(
        f"Pruned SVM package size  : "
        f"{model_size_mb:.4f} MB"
    )


    # ========================================================
    # Save detailed result
    # ========================================================

    result_row = {

        "Method":
            "Proposed",

        "N_test":
            len(y),

        "Batch_size":
            BATCH_SIZE,

        "Warmup_iterations":
            WARMUP_ITERS,

        "Timing_iterations":
            TIMING_ITERS,

        "Full_feature_dimension":
            len(feature_names),

        "Pruned_feature_dimension":
            len(pruned_feature_names),

        "Accuracy_percent":
            accuracy * 100.0,

        "Macro_F1_percent":
            macro_f1 * 100.0,

        "Feature_extraction_ms_per_image":
            feature_ms,

        "Feature_extraction_FPS":
            feat_result["fps"],

        "Preprocessing_ms_per_image":
            preprocessing_ms,

        "Preprocessing_SD_ms":
            prep_result["std_ms"],

        "SVM_inference_ms_per_image":
            svm_ms,

        "SVM_inference_SD_ms":
            svm_result["std_ms"],

        "SVM_inference_median_ms":
            svm_result["median_ms"],

        "SVM_inference_P95_ms":
            svm_result["p95_ms"],

        "SVM_inference_FPS":
            svm_result["fps"],

        "End_to_end_ms_per_image":
            total_ms,

        "End_to_end_FPS":
            total_fps,

        "Feature_extraction_peak_RAM_MB":
            feat_result[
                "peak_rss_mb"
            ],

        "Feature_extraction_increment_RAM_MB":
            feat_result[
                "peak_increment_mb"
            ],

        "SVM_inference_peak_RAM_MB":
            svm_result[
                "peak_rss_mb"
            ],

        "Model_size_MB":
            model_size_mb,
    }


    result_df = pd.DataFrame(
        [result_row]
    )


    csv_path = (
        SAVE_DIR
        / "Proposed_efficiency_results.csv"
    )


    result_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Save readable TXT report
    # ========================================================

    report_path = (
        SAVE_DIR
        / "Proposed_efficiency_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Reviewer 2 - Comment 7\n"
        )

        f.write(
            "Proposed method efficiency benchmark\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"N test = {len(y)}\n"
        )

        f.write(
            f"Batch size = "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Warm-up = "
            f"{WARMUP_ITERS}\n"
        )

        f.write(
            f"Timed iterations = "
            f"{TIMING_ITERS}\n\n"
        )

        f.write(
            f"Accuracy = "
            f"{accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Macro-F1 = "
            f"{macro_f1 * 100:.2f}%\n\n"
        )

        f.write(
            f"Feature extraction = "
            f"{feature_ms:.4f} ms/image\n"
        )

        f.write(
            f"Feature preprocessing = "
            f"{preprocessing_ms:.4f} ms/image\n"
        )

        f.write(
            f"Pure SVM inference = "
            f"{svm_ms:.4f} ms/image\n"
        )

        f.write(
            f"End-to-end latency = "
            f"{total_ms:.4f} ms/image\n"
        )

        f.write(
            f"Pure SVM FPS = "
            f"{svm_result['fps']:.2f}\n"
        )

        f.write(
            f"End-to-end FPS = "
            f"{total_fps:.2f}\n\n"
        )

        f.write(
            f"Feature extraction "
            f"peak RAM = "
            f"{feat_result['peak_rss_mb']:.2f} MB\n"
        )

        f.write(
            f"SVM inference peak RAM = "
            f"{svm_result['peak_rss_mb']:.2f} MB\n"
        )

        f.write(
            f"Model size = "
            f"{model_size_mb:.4f} MB\n"
        )


    print(
        "\nSaved CSV:"
    )

    print(
        csv_path
    )


    print(
        "\nSaved report:"
    )

    print(
        report_path
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
