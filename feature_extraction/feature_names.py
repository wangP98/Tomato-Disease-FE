# -*- coding: utf-8 -*-
"""Canonical feature names and dimensionalities."""

from feature_extraction.color_features import BINS_COLOR
from feature_extraction.lbp_features import BINS_LBP
from feature_extraction.lesion_features import BINS_LESION_LBP

COLOR_DIM = 192
LBP_DIM = 70
SHAPE_DIM = 12
LESION_DIM = 22
TOTAL_DIM = 296


def build_feature_names(
    bins_color: int = BINS_COLOR,
    bins_lbp: int = BINS_LBP,
    use_color: bool = True,
    use_lbp: bool = True,
    use_shape: bool = True,
    use_lesion: bool = True,
):
    """Build feature names in the same order as the extraction pipeline."""
    names = []

    if use_color:
        for color_space in ["HSV", "Lab"]:
            for channel in range(3):
                names += [
                    f"{color_space}_ch{channel}_hist_bin{i}"
                    for i in range(bins_color)
                ]

    if use_lbp:
        names += [f"LBP_feat_{i}" for i in range(bins_lbp)]

    if use_shape:
        names += [
            "area_ratio",
            "rect_ratio",
            "solidity",
            "eccentricity",
            "extent",
            "hu1",
            "hu2",
            "hu3",
            "hu4",
            "hu5",
            "hu6",
            "hu7",
        ]

    if use_lesion:
        names += [
            "lesion_ratio",
            "lesion_count",
            "lesion_big_count",
            "lesion_small_count",
            "lesion_edge_ratio",
            "lesion_inner_ratio",
        ]
        names += [
            f"lesion_lbp_bin{i}"
            for i in range(BINS_LESION_LBP)
        ]

    return names
