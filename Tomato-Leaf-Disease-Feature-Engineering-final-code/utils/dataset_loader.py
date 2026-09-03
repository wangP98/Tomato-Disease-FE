# -*- coding: utf-8 -*-
"""PlantVillage split loading and optional leaf-mask caching."""

from pathlib import Path

import cv2
import numpy as np

from segmentation.leaf_segmentation import segment_leaf
from feature_extraction.extract_all_features import extract_all_features
from utils.io_utils import imread_unicode, imwrite_unicode

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _mask_cache_path(
    img_path: Path,
    mask_root: Path,
    common_root: Path = None,
):
    if common_root is not None:
        try:
            rel_path = img_path.relative_to(common_root)
        except ValueError:
            rel_path = Path(img_path.name)
    else:
        rel_path = Path(img_path.name)

    return mask_root / rel_path.parent / f"{rel_path.stem}_mask.png"


def extract_features_for_image(
    img_bgr,
    img_path=None,
    mask_root=Path("masks"),
    common_root=None,
    use_color=True,
    use_lbp=True,
    use_shape=True,
    use_lesion=True,
):
    """
    Extract features for one image, reusing a cached leaf mask when available.
    """
    mask_root = Path(mask_root)
    mask_path = None

    if img_path is not None:
        img_path = Path(img_path)
        common_root = Path(common_root) if common_root is not None else None
        mask_path = _mask_cache_path(
            img_path,
            mask_root,
            common_root,
        )

    if mask_path is not None and mask_path.exists():
        leaf_mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if leaf_mask is None:
            leaf_mask, _ = segment_leaf(img_bgr)
            imwrite_unicode(mask_path, leaf_mask)
    else:
        leaf_mask, _ = segment_leaf(img_bgr)
        if mask_path is not None:
            imwrite_unicode(mask_path, leaf_mask)

    return extract_all_features(
        img_bgr,
        leaf_mask=leaf_mask,
        use_color=use_color,
        use_lbp=use_lbp,
        use_shape=use_shape,
        use_lesion=use_lesion,
    )


def load_split_features(
    root_dir,
    class_to_id,
    mask_root=Path("masks"),
    common_root=None,
    use_color=True,
    use_lbp=True,
    use_shape=True,
    use_lesion=True,
):
    """
    Extract X and y for a directory split organized as:
        split/class_name/image.ext
    """
    root = Path(root_dir)
    class_names = list(class_to_id.keys())

    extra = [
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name not in class_to_id
    ]
    if extra:
        raise ValueError(
            f"{root} contains undeclared class directories: {extra}"
        )

    X, y = [], []

    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() not in IMG_EXTS:
                continue

            image = imread_unicode(image_path)
            if image is None:
                continue

            features = extract_features_for_image(
                img_bgr=image,
                img_path=image_path,
                mask_root=mask_root,
                common_root=common_root,
                use_color=use_color,
                use_lbp=use_lbp,
                use_shape=use_shape,
                use_lesion=use_lesion,
            )

            X.append(features)
            y.append(class_to_id[class_name])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
    )
