# -*- coding: utf-8 -*-
"""Multi-scale global + local-grid LBP descriptors."""

import numpy as np
from skimage.feature import local_binary_pattern

LBP_P = 8
LBP_GLOBAL_RADII = (1.0, 2.0, 4.0)
LBP_GRID_N = 2
LBP_CODE_BINS = LBP_P + 2
BINS_LBP = int(
    LBP_CODE_BINS * (len(LBP_GLOBAL_RADII) + LBP_GRID_N ** 2)
)


def lbp_texture_features_multi_local(
    gray: np.ndarray,
    leaf_mask: np.ndarray,
    P: int = LBP_P,
    R_list=LBP_GLOBAL_RADII,
    grid_n: int = LBP_GRID_N,
) -> np.ndarray:
    """
    Extract global multi-scale and local 2x2-grid uniform-LBP histograms.

    With P=8, radii=(1,2,4), and a 2x2 local grid:
        10 x (3 + 4) = 70 dimensions.
    """
    roi_mask = leaf_mask > 0
    height, width = gray.shape
    feats = []
    code_bins = int(P + 2)

    # Global LBP at three scales.
    for radius in R_list:
        lbp = local_binary_pattern(
            gray,
            P=P,
            R=radius,
            method="uniform",
        ).astype(np.int32)

        vals = lbp[roi_mask].ravel()
        vals = np.clip(vals, 0, code_bins)

        hist, _ = np.histogram(
            vals,
            bins=code_bins,
            range=(0, code_bins),
        )
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)

    # Local 2x2 grid using the first radius.
    radius0 = R_list[0]
    lbp_local = local_binary_pattern(
        gray,
        P=P,
        R=radius0,
        method="uniform",
    ).astype(np.int32)

    cell_h = height // grid_n
    cell_w = width // grid_n

    for gy in range(grid_n):
        for gx in range(grid_n):
            y0 = gy * cell_h
            y1 = height if gy == grid_n - 1 else (gy + 1) * cell_h
            x0 = gx * cell_w
            x1 = width if gx == grid_n - 1 else (gx + 1) * cell_w

            cell_mask = roi_mask[y0:y1, x0:x1]

            if not cell_mask.any():
                hist = np.zeros(code_bins, dtype=np.float32)
            else:
                vals = lbp_local[y0:y1, x0:x1][cell_mask].ravel()
                vals = np.clip(vals, 0, code_bins)
                hist, _ = np.histogram(
                    vals,
                    bins=code_bins,
                    range=(0, code_bins),
                )
                hist = hist.astype(np.float32)
                hist = hist / (hist.sum() + 1e-6)

            feats.append(hist)

    features = np.concatenate(feats, axis=0).astype(np.float32)

    expected_dim = int(code_bins * (len(R_list) + grid_n ** 2))
    if features.shape[0] != expected_dim:
        raise RuntimeError(
            f"Unexpected LBP dimension: "
            f"{features.shape[0]} != {expected_dim}"
        )

    return features
