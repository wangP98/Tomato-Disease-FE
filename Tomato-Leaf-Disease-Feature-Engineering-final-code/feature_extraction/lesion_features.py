# -*- coding: utf-8 -*-
"""
22-dimensional lesion feature group:
1 severity + 3 granularity + 2 LSL + 16 intralesion LBP.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from segmentation.lesion_segmentation import segment_necrotic_lesions
from feature_extraction.lsl_features import lsl_features

BIG_LESION_RATIO_THRESHOLD = 0.02
BINS_LESION_LBP = 16
LESION_LBP_P = 8
LESION_LBP_R = 1

LESION_BASIC_DIM = 4
LSL_DIM = 2
LESION_LBP_DIM = 16
LESION_TOTAL_DIM = LESION_BASIC_DIM + LSL_DIM + LESION_LBP_DIM


def lesion_severity_granularity_features(
    necrotic_mask: np.ndarray,
    leaf_mask: np.ndarray,
    big_ratio_thr: float = BIG_LESION_RATIO_THRESHOLD,
) -> np.ndarray:
    """
    Extract severity and lesion-count/granularity descriptors.

    Returns
    -------
    [lesion_ratio, lesion_count, big_count, small_count]
    """
    lesion = (necrotic_mask > 0).astype(np.uint8)
    leaf = (leaf_mask > 0).astype(np.uint8)

    leaf_area = float(leaf.sum()) + 1e-6
    lesion_area = float(lesion.sum())
    lesion_ratio = lesion_area / leaf_area

    num_labels, labels = cv2.connectedComponents(lesion)
    lesion_count = num_labels - 1

    big_count = 0
    small_count = 0

    for label_id in range(1, num_labels):
        area_i = float((labels == label_id).sum())
        if area_i / leaf_area >= big_ratio_thr:
            big_count += 1
        else:
            small_count += 1

    return np.array(
        [
            lesion_ratio,
            float(lesion_count),
            float(big_count),
            float(small_count),
        ],
        dtype=np.float32,
    )


def lesion_lbp_features(
    gray: np.ndarray,
    necrotic_mask: np.ndarray,
    P: int = LESION_LBP_P,
    R: float = LESION_LBP_R,
    bins: int = BINS_LESION_LBP,
) -> np.ndarray:
    """
    Compute a 16-bin uniform-LBP histogram only within the lesion region.
    """
    mask = necrotic_mask > 0

    if mask.sum() == 0:
        return np.zeros(bins, dtype=np.float32)

    lbp = local_binary_pattern(
        gray,
        P=P,
        R=R,
        method="uniform",
    )
    lbp = lbp[mask]

    hist, _ = np.histogram(
        lbp,
        bins=bins,
        range=(0, P + 2),
        density=True,
    )
    return hist.astype(np.float32)


def extract_lesion_features(
    gray: np.ndarray,
    leaf_mask: np.ndarray,
    return_mask: bool = False,
    return_details: bool = False,
):
    """
    Extract the complete 22-dimensional lesion feature group.

    Feature order
    -------------
    0  lesion_ratio
    1  lesion_count
    2  lesion_big_count
    3  lesion_small_count
    4  lesion_edge_ratio (LSL R_edge)
    5  lesion_inner_ratio (LSL R_inner)
    6:22  intralesion LBP histogram
    """
    necrotic_mask, t_dark = segment_necrotic_lesions(
        gray,
        leaf_mask,
    )

    basic = lesion_severity_granularity_features(
        necrotic_mask,
        leaf_mask,
    )

    lsl = lsl_features(
        leaf_mask,
        necrotic_mask,
    )

    lesion_lbp = lesion_lbp_features(
        gray,
        necrotic_mask,
        P=LESION_LBP_P,
        R=LESION_LBP_R,
        bins=BINS_LESION_LBP,
    )

    features = np.concatenate(
        [basic, lsl, lesion_lbp],
        axis=0,
    ).astype(np.float32)

    if features.shape[0] != LESION_TOTAL_DIM:
        raise RuntimeError(
            f"Unexpected lesion-feature dimension: "
            f"{features.shape[0]} != {LESION_TOTAL_DIM}"
        )

    if return_details:
        details = {
            "t_dark": t_dark,
            "lesion_ratio": float(basic[0]),
            "lesion_count": int(basic[1]),
            "lesion_big_count": int(basic[2]),
            "lesion_small_count": int(basic[3]),
            "r_edge": float(lsl[0]),
            "r_inner": float(lsl[1]),
        }
        if return_mask:
            return features, necrotic_mask, details
        return features, details

    if return_mask:
        return features, necrotic_mask

    return features
