# -*- coding: utf-8 -*-
"""Single-image feature extraction and 296-D feature assembly."""

from pathlib import Path

import cv2
import numpy as np

from segmentation.leaf_segmentation import segment_leaf
from feature_extraction.color_features import color_hist_features
from feature_extraction.lbp_features import lbp_texture_features_multi_local
from feature_extraction.shape_features import shape_features
from feature_extraction.lesion_features import extract_lesion_features
from feature_extraction.feature_names import build_feature_names


def extract_all_features(
    img_bgr: np.ndarray,
    leaf_mask: np.ndarray = None,
    use_color: bool = True,
    use_lbp: bool = True,
    use_shape: bool = True,
    use_lesion: bool = True,
    return_details: bool = False,
):
    """
    Extract the requested feature groups from a single BGR image.

    With all groups enabled, the output dimension is:
        192 color + 70 LBP + 12 shape + 22 lesion = 296.
    """
    if leaf_mask is None:
        leaf_mask, triangle_threshold = segment_leaf(img_bgr)
    else:
        triangle_threshold = None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    feat_list = []
    details = {
        "triangle_threshold": triangle_threshold,
    }

    if use_color:
        feat_list.append(
            color_hist_features(img_bgr, leaf_mask)
        )

    if use_lbp:
        feat_list.append(
            lbp_texture_features_multi_local(gray, leaf_mask)
        )

    if use_shape:
        feat_list.append(
            shape_features(leaf_mask)
        )

    if use_lesion:
        lesion_features, necrotic_mask, lesion_details = extract_lesion_features(
            gray,
            leaf_mask,
            return_mask=True,
            return_details=True,
        )
        feat_list.append(lesion_features)
        details["necrotic_mask"] = necrotic_mask
        details["lesion"] = lesion_details

    if not feat_list:
        raise ValueError("At least one feature group must be enabled.")

    features = np.concatenate(
        feat_list,
        axis=0,
    ).astype(np.float32)

    feature_names = build_feature_names(
        use_color=use_color,
        use_lbp=use_lbp,
        use_shape=use_shape,
        use_lesion=use_lesion,
    )

    if len(features) != len(feature_names):
        raise RuntimeError(
            f"Feature vector/name mismatch: "
            f"{len(features)} != {len(feature_names)}"
        )

    if return_details:
        details["leaf_mask"] = leaf_mask
        details["feature_names"] = feature_names
        return features, details

    return features


extract_features = extract_all_features
