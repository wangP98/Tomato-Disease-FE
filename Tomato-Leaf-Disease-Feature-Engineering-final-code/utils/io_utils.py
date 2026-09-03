# -*- coding: utf-8 -*-
"""File I/O helpers, including Unicode-path image reading."""

from pathlib import Path
import os

import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Read an image from a path that may contain non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image) -> bool:
    """Write an image to a path that may contain non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix
    ok, encoded = cv2.imencode(ext, image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)
