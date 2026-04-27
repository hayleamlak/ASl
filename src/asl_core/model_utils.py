import json
from pathlib import Path

import joblib
import numpy as np


def load_labels(labels_path: str) -> list[str]:
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            sorted_items = sorted(data.items(), key=lambda item: int(item[0]))
            return [str(v) for _, v in sorted_items]
        raise ValueError("Unsupported JSON label format. Use a list or index-keyed object.")

    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("Labels file is empty.")
    return labels


def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(path)


def predict_label(model, features: np.ndarray) -> tuple[str, float]:
    sample = features.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample)[0]
        idx = int(np.argmax(proba))
        return str(model.classes_[idx]), float(proba[idx])

    pred = model.predict(sample)[0]
    return str(pred), 1.0
