import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np


@dataclass
class ModelBundle:
    model: object
    vectorizer: object
    label_map: Dict[int, str]


class Predictor:
    def __init__(self) -> None:
        self._bundle: Optional[ModelBundle] = None
        self.model_dir: Optional[str] = None
        self.model_version: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def load(self, model_dir: str) -> None:
        model_path = Path(model_dir)
        model = joblib.load(model_path / "model.pkl")
        vectorizer = joblib.load(model_path / "vectorizer.pkl")
        with (model_path / "label_map.json").open("r", encoding="utf-8") as handle:
            raw_map = json.load(handle)
        label_map = {int(key): value for key, value in raw_map.items()}
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.model_version = metadata.get("model_version")
        self._bundle = ModelBundle(model=model, vectorizer=vectorizer, label_map=label_map)
        self.model_dir = str(model_path)

    def predict(
        self, texts: List[str], top_k: int = 3, min_confidence: float = 0.55
    ) -> List[Dict[str, object]]:
        if not self._bundle:
            raise RuntimeError("Model not loaded")
        vectorizer = self._bundle.vectorizer
        model = self._bundle.model
        label_map = self._bundle.label_map
        matrix = vectorizer.transform(texts)
        probabilities = model.predict_proba(matrix)
        num_classes = probabilities.shape[1]
        k = min(top_k, num_classes)
        results: List[Dict[str, object]] = []
        for row in probabilities:
            top_indices = np.argsort(row)[::-1][:k]
            alternatives = [
                {
                    "label": label_map[int(idx)],
                    "confidence": float(row[int(idx)]),
                }
                for idx in top_indices
            ]
            top_label = alternatives[0]["label"]
            top_confidence = alternatives[0]["confidence"]
            needs_human = top_confidence < min_confidence
            label = "human_review" if needs_human else top_label
            results.append(
                {
                    "label": label,
                    "confidence": top_confidence,
                    "alternatives": alternatives,
                    "needs_human": needs_human,
                }
            )
        return results
