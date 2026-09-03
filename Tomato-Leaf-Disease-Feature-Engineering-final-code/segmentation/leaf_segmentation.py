# -*- coding: utf-8 -*-
"""
Core ExG-Triangle leaf segmentation used in the tomato leaf disease study.
"""

import cv2
import numpy as np
from skimage.filters import threshold_triangle

GAUSSIAN_KERNEL = (5, 5)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

YELLOW_H_MIN = 15
YELLOW_H_MAX = 50
YELLOW_S_MIN = 30
YELLOW_SCORE_WEIGHT = 0.7
YELLOW_INDICATOR_WEIGHT = 0.3
YELLOW_THRESHOLD_MIN = 0.35
YELLOW_THRESHOLD_SCALE = 0.75

BACKGROUND_BORDER = 14
MAHALANOBIS_K = 2.2

MORPH_KERNEL_SIZE = 5
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 2


def norm01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-6)


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Retain the largest 8-connected foreground component."""
    m = (mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    max_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.uint8(labels == max_id) * 255


def morph_clean(
    mask: np.ndarray,
    ksize: int = MORPH_KERNEL_SIZE,
    open_iter: int = MORPH_OPEN_ITER,
    close_iter: int = MORPH_CLOSE_ITER,
) -> np.ndarray:
    """Apply the morphology used by the ExG-Triangle leaf mask."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=open_iter)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=close_iter)
    return out


def clahe_l_channel(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel in Lab color space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    lc = clahe.apply(l)
    labc = cv2.merge([lc, a, b])
    return cv2.cvtColor(labc, cv2.COLOR_LAB2BGR)


def background_mask_lab(
    bgr: np.ndarray,
    border: int = BACKGROUND_BORDER,
    k: float = MAHALANOBIS_K,
) -> np.ndarray:
    """
    Identify background-like pixels from the Lab distribution of the image border.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)

    border_mask = np.zeros((h, w), np.uint8)
    border_mask[:border, :] = 1
    border_mask[-border:, :] = 1
    border_mask[:, :border] = 1
    border_mask[:, -border:] = 1

    bg = lab[border_mask.reshape(-1).astype(bool)]
    mu = bg.mean(axis=0)
    cov = np.cov(bg.T) + np.eye(3) * 1e-6
    inv = np.linalg.inv(cov)

    d = lab - mu
    md2 = np.einsum("ij,jk,ik->i", d, inv, d).reshape(h, w)
    return md2 < (k ** 2)


def segment_leaf_exg_triangle(
    bgr: np.ndarray,
    return_details: bool = False,
):
    """
    Segment a tomato leaf using the ExG-Triangle procedure.

    Returns
    -------
    leaf_mask : uint8 ndarray
        Binary mask with foreground=255.
    triangle_threshold : float
        Per-image adaptive Triangle threshold.

    If ``return_details=True``, the second return value is a dict containing
    both the Triangle threshold and the yellow-recovery threshold.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Input must be a valid HxWx3 BGR image.")

    # 1) ExG response
    b, g, r = cv2.split(bgr.astype(np.float32))
    exg = 2.0 * g - r - b

    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    exg_u8 = cv2.GaussianBlur(exg_u8, GAUSSIAN_KERNEL, 0)

    triangle_threshold = float(threshold_triangle(exg_u8))
    m_exg = (exg_u8 > triangle_threshold).astype(np.uint8) * 255

    # 2) Yellow/chlorotic-region recovery
    bgrc = clahe_l_channel(bgr)

    lab = cv2.cvtColor(bgrc, cv2.COLOR_BGR2LAB)
    _, _, bch = cv2.split(lab)

    hsv = cv2.cvtColor(bgrc, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)

    yellow_indicator = (
        (h >= YELLOW_H_MIN)
        & (h <= YELLOW_H_MAX)
        & (s > YELLOW_S_MIN)
    ).astype(np.uint8)

    b_norm = norm01(bch)
    s_norm = norm01(s)

    yellow_score = (
        YELLOW_SCORE_WEIGHT * b_norm * s_norm
        + YELLOW_INDICATOR_WEIGHT * yellow_indicator
    )

    yellow_threshold = max(
        YELLOW_THRESHOLD_MIN,
        float(YELLOW_THRESHOLD_SCALE * b_norm.mean()),
    )

    m_yellow = (yellow_score > yellow_threshold).astype(np.uint8) * 255

    # 3) Border-derived background removal
    bg_like = background_mask_lab(
        bgrc,
        border=BACKGROUND_BORDER,
        k=MAHALANOBIS_K,
    )

    m = cv2.bitwise_or(m_exg, m_yellow)
    m[bg_like] = 0

    # 4) Morphology + largest 8-connected component
    m = morph_clean(
        m,
        ksize=MORPH_KERNEL_SIZE,
        open_iter=MORPH_OPEN_ITER,
        close_iter=MORPH_CLOSE_ITER,
    )
    leaf_mask = largest_cc(m)

    if return_details:
        return leaf_mask, {
            "triangle_threshold": triangle_threshold,
            "yellow_threshold": yellow_threshold,
        }

    return leaf_mask, triangle_threshold


segment_leaf = segment_leaf_exg_triangle
