# -*- coding: utf-8 -*-
"""Lesion Spatial Logic (LSL) descriptor."""

import cv2
import numpy as np

LSL_KERNEL_SIZE = 5
LSL_DIM = 2


def lsl_features(
    leaf_mask: np.ndarray,
    necrotic_mask: np.ndarray,
    kernel_size: int = LSL_KERNEL_SIZE,
) -> np.ndarray:
    """
    Compute the two-dimensional LSL descriptor [R_edge, R_inner].

    The leaf interior is obtained by one 5x5 erosion. The marginal region is
    the difference between the original leaf mask and the eroded interior.
    Both lesion ratios are normalized by the total leaf area.
    """
    leaf = (leaf_mask > 0).astype(np.uint8)
    lesion = (necrotic_mask > 0).astype(np.uint8)

    leaf_area = float(leaf.sum()) + 1e-6

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    inner_mask = cv2.erode(
        leaf,
        kernel,
        iterations=1,
    )

    edge_mask = leaf - inner_mask
    edge_mask = (edge_mask > 0).astype(np.uint8)

    edge_lesion = lesion * edge_mask
    inner_lesion = lesion * inner_mask

    edge_ratio = float(edge_lesion.sum()) / leaf_area
    inner_ratio = float(inner_lesion.sum()) / leaf_area

    return np.array(
        [edge_ratio, inner_ratio],
        dtype=np.float32,
    )
