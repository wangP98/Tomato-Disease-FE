# -*- coding: utf-8 -*-
"""
Optional feature-space augmentation used for the pruned SVM training set.

This module does NOT apply feature-group weights.
"""

import numpy as np


def get_color_feature_indices(feature_names):
    """Return indices of HSV/Lab color features."""
    return np.asarray(
        [
            i
            for i, name in enumerate(feature_names)
            if name.startswith("HSV_") or name.startswith("Lab_")
        ],
        dtype=int,
    )


def augment_features(
    X,
    y,
    n_aug=1,
    noise_std=0.01,
    color_idx=None,
    bright_range=(0.9, 1.1),
    random_state=42,
):
    """
    Augment standardized features using Gaussian noise and color-feature scaling.

    Parameters are preserved from the supplied training implementation.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(random_state)

    X_list = [X]
    y_list = [y]

    n_samples, n_features = X.shape

    for _ in range(n_aug):
        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(n_samples, n_features),
        )
        X_aug = X + noise

        if color_idx is not None and len(color_idx) > 0:
            scale = rng.uniform(
                bright_range[0],
                bright_range[1],
                size=(n_samples, 1),
            )
            X_aug[:, color_idx] *= scale

        X_list.append(X_aug)
        y_list.append(y)

    return np.vstack(X_list), np.concatenate(y_list)
