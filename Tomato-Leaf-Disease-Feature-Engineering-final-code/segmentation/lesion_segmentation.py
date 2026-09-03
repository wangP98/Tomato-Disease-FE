# -*- coding: utf-8 -*-
"""
Necrotic/dark lesion-mask extraction inside the segmented leaf.
"""

import cv2
import numpy as np

LESION_THRESHOLD_PERCENTILE = 30.0
LESION_MORPH_KERNEL_SIZE = 3


def segment_necrotic_lesions(
    gray: np.ndarray,
    leaf_mask: np.ndarray,
    percentile: float = LESION_THRESHOLD_PERCENTILE,
    kernel_size: int = LESION_MORPH_KERNEL_SIZE,
):
    """
    Extract the necrotic/dark-region mask M_nec.

    The adaptive threshold is the requested percentile of grayscale intensities
    inside the segmented leaf. The original implementation uses the 30th
    percentile. A 3x3 opening followed by a 3x3 closing is then applied.

    Parameters
    ----------
    gray : ndarray
        Grayscale image.
    leaf_mask : ndarray
        Binary leaf mask; any positive value is treated as foreground.
    percentile : float
        Grayscale percentile used as T_dark.
    kernel_size : int
        Square morphology kernel size.

    Returns
    -------
    necrotic_mask : uint8 ndarray
        Binary 0/1 mask used by the lesion and LSL descriptors.
    t_dark : float
        Adaptive grayscale threshold.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel image.")

    leaf_fg = leaf_mask > 0
    leaf_pixels = gray[leaf_fg]

    if leaf_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8), float("nan")

    t_dark = float(np.percentile(leaf_pixels, percentile))

    # Preserve the behavior of the original implementation:
    # pixels outside the leaf are set to 255 before inverse thresholding.
    leaf_gray = gray.copy()
    leaf_gray[~leaf_fg] = 255

    _, lesion = cv2.threshold(
        leaf_gray,
        t_dark,
        255,
        cv2.THRESH_BINARY_INV,
    )
    lesion = cv2.bitwise_and(
        lesion,
        lesion,
        mask=leaf_fg.astype(np.uint8),
    )

    necrotic_mask = (lesion > 0).astype(np.uint8)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        necrotic_mask = cv2.morphologyEx(
            necrotic_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
        necrotic_mask = cv2.morphologyEx(
            necrotic_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

    return necrotic_mask.astype(np.uint8), t_dark


segment_lesions = segment_necrotic_lesions
