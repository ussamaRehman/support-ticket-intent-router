import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_test_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        "refund my card",
        "reset my password",
        "refund charged twice",
        "change account email",
    ]
    labels = [0, 1, 0, 1]
    label_names = ["billing", "account"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=200)
    model.fit(X, labels)

    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")

    label_map = {str(idx): label for idx, label in enumerate(label_names)}
    with (model_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    metadata = {
        "model_version": "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return model_dir


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return build_test_model(tmp_path)
