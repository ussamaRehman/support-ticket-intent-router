import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from training.data import load_sample_split

MODEL_VERSION = "0.1.0"
DEFAULT_MODEL_DIR = Path("artifacts") / f"model_{MODEL_VERSION}"


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top_indices = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    hits = [true in row for true, row in zip(y_true, top_indices)]
    return float(np.mean(hits)) if hits else 0.0


def build_metadata(
    metrics_summary: Dict[str, float], seed: int, dataset_name: str
) -> Dict[str, object]:
    return {
        "dataset": dataset_name,
        "seed": seed,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "metrics_summary": metrics_summary,
    }


def main() -> None:
    model_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
    ).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    seed = 42
    dataset_name = "sample_v1"

    X_train, X_test, y_train, y_test = load_sample_split(random_state=seed)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train_enc)

    X_test_vec = vectorizer.transform(X_test)
    test_proba = model.predict_proba(X_test_vec)
    test_preds = model.predict(X_test_vec)
    metrics_summary = {
        "macro_f1": float(f1_score(y_test_enc, test_preds, average="macro")),
        "top_k_accuracy": float(top_k_accuracy(y_test_enc, test_proba, k=3)),
    }

    label_map = {str(idx): label for idx, label in enumerate(label_encoder.classes_)}

    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    with (model_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    metadata = build_metadata(metrics_summary=metrics_summary, seed=seed, dataset_name=dataset_name)
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved artifacts to {model_dir}")


if __name__ == "__main__":
    main()
