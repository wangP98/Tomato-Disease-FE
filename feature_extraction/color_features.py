# -*- coding: utf-8 -*-
"""HSV + Lab color-histogram descriptors."""

import cv2
import numpy as np

BINS_COLOR = 32


def color_hist_features(
    bgr: np.ndarray,
    leaf_mask: np.ndarray,
    bins: int = BINS_COLOR,
) -> np.ndarray:
    """
    Extract normalized HSV and Lab histograms within the leaf mask.

    Dimension with the study setting:
        2 color spaces x 3 channels x 32 bins = 192.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    mask = (leaf_mask > 0).astype(np.uint8)
    feats = []

    ranges_hsv = [(0, 180), (0, 256), (0, 256)]
    for channel, value_range in enumerate(ranges_hsv):
        hist = cv2.calcHist(
            [hsv],
            [channel],
            mask,
            [bins],
            list(value_range),
        ).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)

    for channel in range(3):
        hist = cv2.calcHist(
            [lab],
            [channel],
            mask,
            [bins],
            [0, 256],
        ).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)

    features = np.concatenate(feats, axis=0).astype(np.float32)
    expected_dim = 6 * bins
    if features.shape[0] != expected_dim:
        raise RuntimeError(
            f"Unexpected color-feature dimension: "
            f"{features.shape[0]} != {expected_dim}"
        )
    return features
