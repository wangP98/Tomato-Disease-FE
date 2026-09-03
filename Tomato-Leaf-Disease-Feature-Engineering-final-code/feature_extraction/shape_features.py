# -*- coding: utf-8 -*-
"""Whole-leaf morphology descriptors."""

import cv2
import numpy as np
from skimage.measure import regionprops

SHAPE_DIM = 12


def shape_features(leaf_mask: np.ndarray) -> np.ndarray:
    """
    Extract 12 whole-leaf shape descriptors:
    area ratio, bounding-rectangle ratio, solidity, eccentricity, extent,
    and seven signed-log Hu moments.
    """
    mask = (leaf_mask > 0).astype(np.uint8)
    height, width = mask.shape

    area = float(mask.sum())
    area_ratio = area / (height * width + 1e-6)

    contours = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = contours[0] if len(contours) == 2 else contours[1]

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(contour)
        rect_area = float(w * h)
        rect_ratio = area / (rect_area + 1e-6)
    else:
        rect_ratio = 0.0

    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = np.where(hu == 0, 1e-6, hu)
    hu = -np.sign(hu) * np.log10(np.abs(hu))

    props = regionprops(mask)
    if props:
        rp = max(props, key=lambda p: p.area)
        solidity = float(rp.solidity or 0.0)
        eccentricity = float(rp.eccentricity or 0.0)
        extent = float(rp.extent or 0.0)
    else:
        solidity = 0.0
        eccentricity = 0.0
        extent = 0.0

    features = np.array(
        [
            area_ratio,
            rect_ratio,
            solidity,
            eccentricity,
            extent,
            *hu,
        ],
        dtype=np.float32,
    )

    if features.shape[0] != SHAPE_DIM:
        raise RuntimeError(
            f"Unexpected shape-feature dimension: "
            f"{features.shape[0]} != {SHAPE_DIM}"
        )

    return features
