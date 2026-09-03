# -*- coding: utf-8 -*-
"""Class-mapping helpers."""

import json
from pathlib import Path


def load_class_mapping(json_path):
    """
    Load a JSON file with structure:
        {"class_to_id": {"class_name": 0, ...}}
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    class_to_id = payload["class_to_id"]
    id_to_class = {v: k for k, v in class_to_id.items()}
    class_names = list(class_to_id.keys())

    return class_to_id, id_to_class, class_names
