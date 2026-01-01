import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import sklearn

from training.data import TEST_URL, TRAIN_URL, load_banking77_split

MODEL_VERSION = "0.1.0"
DEFAULT_MODEL_DIR = Path("artifacts") / f"model_{MODEL_VERSION}"


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top_indices = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    hits = [true in row for true, row in zip(y_true, top_indices)]
    return float(np.mean(hits)) if hits else 0.0


def build_metadata(
    metrics_summary: Dict[str, float],
    seed: int,
    dataset_name: str,
    license_name: str,
    dataset_source_urls: Dict[str, str],
    n_train: int,
    n_test: int,
    label_encoding: str,
) -> Dict[str, object]:
    return {
        "dataset": dataset_name,
        "dataset_name": dataset_name,
        "dataset_source_urls": dataset_source_urls,
        "license": license_name,
        "n_train": n_train,
        "n_test": n_test,
        "label_encoding": label_encoding,
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
    dataset_name = "Banking77 (PolyAI task-specific-datasets)"
    license_name = "CC BY 4.0"
    dataset_source_urls = {"train": TRAIN_URL, "test": TEST_URL}
    label_encoding = "sorted_label_names"

    X_train, y_train, X_test, y_test, label_names = load_banking77_split(seed=seed)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    test_proba = model.predict_proba(X_test_vec)
    test_preds = model.predict(X_test_vec)
    metrics_summary = {
        "macro_f1": float(f1_score(y_test, test_preds, average="macro")),
        "top_k_accuracy": float(top_k_accuracy(y_test, test_proba, k=3)),
    }

    label_map = {str(idx): label for idx, label in enumerate(label_names)}

    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    with (model_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    metadata = build_metadata(
        metrics_summary=metrics_summary,
        seed=seed,
        dataset_name=dataset_name,
        license_name=license_name,
        dataset_source_urls=dataset_source_urls,
        n_train=len(X_train),
        n_test=len(X_test),
        label_encoding=label_encoding,
    )
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved artifacts to {model_dir}")


if __name__ == "__main__":
    main()
