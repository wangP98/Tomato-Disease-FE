# Tomato-Disease-FE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22281949.svg)](https://doi.org/10.5281/zenodo.22281949)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Biologically interpretable feature engineering and two-stage feature pruning for tomato leaf disease classification.

This repository implements a tomato leaf disease classification framework based on explicit pathological feature engineering. The pipeline combines leaf segmentation, color, texture, morphology, lesion descriptors, Lesion Spatial Logic (LSL), two-stage feature pruning, and Support Vector Machine (SVM) classification. It also contains KNN and Random Forest comparison code, CNN baseline training, cross-validation, McNemar analysis, segmentation evaluation, ablation studies, and computational-efficiency benchmarks.

## 1. Repository structure

```text
.
├── datasets/
│   ├── class_to_id.json
│   ├── generate_class_mapping.py
│   └── split_dataset.py
│
├── data_splits/
│   └── export_data_splits.py
│
├── segmentation/
│   ├── leaf_segmentation.py
│   └── lesion_segmentation.py
│
├── feature_extraction/
│   ├── color_features.py
│   ├── lbp_features.py
│   ├── shape_features.py
│   ├── lesion_features.py
│   ├── lsl_features.py
│   ├── feature_names.py
│   └── extract_all_features.py
│
├── feature_selection/
│   ├── permutation_importance.py
│   └── pearson_pruning.py
│
├── classification/
│   ├── train_svm.py
│   └── evaluate_svm.py
│
├── baselines/
│   ├── knn_grid_search.py
│   └── random_forest_grid_search.py
│
├── cross_validation/
│   └── five_fold_cv.py
│
├── statistical_analysis/
│   └── mcnemar_test.py
│
├── reproduce/
│   ├── reproduce_ablation.py
│   ├── reproduce_lsl_ablation.py
│   ├── reproduce_cv.py
│   └── reproduce_mcnemar.py
│
├── deep_learning/
│   ├── dataset.py
│   └── train_cnn_baselines.py
│
├── evaluation/
│   └── evaluate_segmentation.py
│
├── benchmarks/
│   ├── svm_efficiency.py
│   ├── cnn_cpu_efficiency.py
│   └── cnn_gpu_efficiency.py
│
├── feature_metadata/
│   └── feature_names.csv
│
├── configs/
│   └── config.yaml
│
├── checkpoints/
│   └── .gitkeep
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## 2. Main pipeline

The handcrafted representation contains **296 features**:

| Feature group | Dimension |
|---|---:|
| Color | 192 |
| Multi-scale LBP | 70 |
| Shape | 12 |
| Lesion | 22 |
| **Total** | **296** |

The 22-dimensional lesion group contains lesion severity/count descriptors, a **2-D Lesion Spatial Logic (LSL)** descriptor, and 16 intralesion LBP dimensions.

The main processing workflow is:

```text
RGB image
  ↓
leaf segmentation
  ↓
necrotic-region extraction
  ↓
296-D pathological feature representation
  ↓
StandardScaler
  ↓
Permutation Importance
  ↓
Pearson correlation pruning
  ↓
Pruned SVM
```

No feature-group weighting is used in the cleaned public implementation.

## 3. Dataset

The experiments use the tomato subset of the **PlantVillage** dataset.

The raw PlantVillage images are not redistributed in this repository. Arrange the images as:

```text
data01/
├── train/
│   ├── Tomato__Target_Spot/
│   ├── Tomato__Tomato_mosaic_virus/
│   ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_healthy/
│   ├── Tomato_Late_blight/
│   ├── Tomato_Leaf_Mold/
│   ├── Tomato_Septoria_leaf_spot/
│   └── Tomato_Spider_mites_Two_spotted_spider_mite/
└── test/
    └── same 10 class folders
```

The canonical class ID mapping is stored in:

```text
datasets/class_to_id.json
```

Do not regenerate the class mapping when reproducing the reported experiments.

## 4. Environment

The manuscript reports the following main software environment:

```text
Python              3.10.18
scikit-learn        1.7.1
OpenCV              4.11
PyTorch             2.7.1+cu126
Operating system    Microsoft Windows 11
```

A pinned environment is provided in `requirements.txt`.

Create an environment and install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The PyTorch entries in `requirements.txt` target CUDA 12.6. For CPU-only use, install the corresponding CPU PyTorch build instead.

## 5. Export the actual dataset split

If `data01/train` and `data01/test` already contain the exact images used by an experiment, export their relative paths rather than creating a new random split:

```bash
python data_splits/export_data_splits.py \
    --data-root data01 \
    --class-map datasets/class_to_id.json \
    --output-dir data_splits
```

This creates:

```text
data_splits/train.txt
data_splits/test.txt
data_splits/dataset_manifest.csv
data_splits/split_summary.csv
```

The separate `datasets/split_dataset.py` utility is provided for creating a new dataset split, not for claiming that a newly generated split is the historical split used in the manuscript.

## 6. Handcrafted feature extraction

The individual feature modules are located in `feature_extraction/`.

The top-level extraction switches are:

```python
use_color=True
use_lbp=True
use_shape=True
use_lesion=True
```

These switches are also used for the feature-group ablation experiments.

## 7. Train Full and Pruned SVMs

The main SVM implementation is:

```text
classification/train_svm.py
```

The fixed classifier configuration is:

```text
kernel       = rbf
C            = 5.0
gamma        = scale
class_weight = balanced
```

Feature pruning uses:

```text
Permutation repeats               = 10
Permutation scoring               = Macro-F1
Permutation-importance threshold  = 1e-4
Pearson |r| threshold              = 0.99
```

Run the training script according to its command-line arguments and local dataset paths.

## 8. Feature ablation

Top-level feature-group ablation is implemented in:

```text
reproduce/reproduce_ablation.py
```

It evaluates:

```text
Full
w/o Color
w/o LBP
w/o Shape
w/o Lesion
```

The dedicated LSL ablation is:

```text
reproduce/reproduce_lsl_ablation.py
```

For `w/o LSL`, only:

```text
lesion_edge_ratio
lesion_inner_ratio
```

are removed, reducing the representation from 296 to 294 dimensions while retaining the remaining lesion descriptors.

## 9. Leakage-free five-fold cross-validation

Run:

```bash
python reproduce/reproduce_cv.py \
    --x-train outputs/features/X_train_raw.npy \
    --y-train outputs/features/y_train.npy \
    --class-map datasets/class_to_id.json \
    --output-dir outputs/five_fold_cv
```

The validation protocol is:

```text
original training set only
  ↓
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  ↓
StandardScaler fitted on fold-training only
  ↓
Full SVM
  ↓
Permutation Importance on fold-training only
  ↓
Pearson pruning on fold-training only
  ↓
fold-specific Pruned SVM
  ↓
held-out fold validation
```

The independent test set is not accessed by the five-fold feature-selection procedure.

## 10. McNemar test

Run:

```bash
python reproduce/reproduce_mcnemar.py \
    --model-file checkpoints/svm_full_and_pruned.joblib \
    --x-test outputs/features/X_test_raw.npy \
    --y-test outputs/features/y_test.npy \
    --output-dir outputs/mcnemar
```

The implementation reports:

- paired correctness counts;
- uncorrected McNemar chi-square statistic;
- exact two-sided McNemar p-value;
- continuity-corrected statistic as supplemental output.

The primary significance level is `alpha = 0.05`.

## 11. KNN and Random Forest comparison

Parameter-search baselines are located in:

```text
baselines/knn_grid_search.py
baselines/random_forest_grid_search.py
```

Both perform hyperparameter selection on the training set using stratified five-fold cross-validation and Macro-F1. The independent test set is evaluated only after parameter selection.

## 12. CNN baseline training

The unified CNN training script supports:

```text
AlexNet
VGG-16
EfficientNet-B0
MobileNetV3-Large
```

Train one model:

```bash
python deep_learning/train_cnn_baselines.py \
    --model alexnet \
    --data-root data01 \
    --class-map datasets/class_to_id.json
```

Other options are:

```text
--model vgg16
--model efficientnet_b0
--model mobilenet_v3_large
--model all
```

The CNN script creates a stratified validation subset from the original training partition. Validation data are used for learning-rate scheduling and checkpoint selection. The independent test set is evaluated only after training is complete.

Default settings are:

```text
Input size       224 × 224
Batch size       16
Epochs           50
Learning rate    1e-4
Optimizer        SGD
Momentum         0.9
Scheduler        ReduceLROnPlateau
Random seed      42
```

Training augmentation includes horizontal flipping, vertical flipping, ±15° random rotation, and mild color jitter.

## 13. Segmentation evaluation

Reference and predicted masks can be evaluated with:

```bash
python evaluation/evaluate_segmentation.py \
    --reference-root reference_masks/masks \
    --prediction-root predicted_masks \
    --output-dir outputs/segmentation
```

The script calculates:

```text
Dice
IoU
Precision
Recall
```

for each image and reports mean ± sample standard deviation.

## 14. Computational-efficiency benchmarks

The repository includes:

```text
benchmarks/svm_efficiency.py
benchmarks/cnn_cpu_efficiency.py
benchmarks/cnn_gpu_efficiency.py
```

CNN checkpoints can be placed under:

```text
checkpoints/
├── AlexNet/
├── VGG16/
├── EfficientNet-B0/
└── MobileNetV3-Large/
```

The benchmark scripts use repository-relative paths rather than private workstation paths.

## 15. Reproducibility files still tied to the final experiment archive

The code can be used independently, but exact numerical reproduction of a specific archived manuscript run additionally requires the corresponding experiment artifacts, such as:

```text
final selected-feature list
saved SVM model/scaler
actual train/test manifests
actual five-fold manifests
reference masks
final prediction/result CSV files
CNN checkpoints
```

These artifacts should be archived together with the final Zenodo release when available. They should not be reconstructed or fabricated solely to match manuscript tables.

## 16. Citation

Citation metadata are provided in `CITATION.cff`.

Once the final article DOI and Zenodo DOI are available, update `CITATION.cff` and this README with the permanent identifiers.

## 17. License

This source code is released under the MIT License. See `LICENSE`.
