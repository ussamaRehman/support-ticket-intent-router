import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from training.data import load_sample_split

DEFAULT_MODEL_DIR = Path("artifacts") / "model_0.1.0"
DEFAULT_REPORT_DIR = Path("reports")


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top_indices = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    hits = [true in row for true, row in zip(y_true, top_indices)]
    return float(np.mean(hits)) if hits else 0.0


def main() -> None:
    model_dir = Path(DEFAULT_MODEL_DIR).resolve()
    report_dir = Path(DEFAULT_REPORT_DIR).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_dir / "model.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    with (model_dir / "label_map.json").open("r", encoding="utf-8") as handle:
        raw_map: Dict[str, str] = json.load(handle)
    label_map = {int(key): value for key, value in raw_map.items()}
    labels = [label_map[idx] for idx in sorted(label_map.keys())]
    label_to_id = {label: idx for idx, label in label_map.items()}

    X_train, X_test, y_train, y_test = load_sample_split(random_state=42)
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)
    y_test_ids = [label_to_id[label] for label in y_test]

    report = classification_report(
        y_test_ids,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "macro_f1": float(f1_score(y_test_ids, y_pred, average="macro")),
        "top_k_accuracy": float(top_k_accuracy(np.array(y_test_ids), y_proba, k=3)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "support_total": int(report["macro avg"]["support"]),
    }

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    cm = confusion_matrix(y_test_ids, y_pred, labels=list(range(len(labels))))
    cm_payload = {"labels": labels, "matrix": cm.tolist()}
    with (report_dir / "confusion_matrix.json").open("w", encoding="utf-8") as handle:
        json.dump(cm_payload, handle, indent=2)

    print(f"Saved reports to {report_dir}")


if __name__ == "__main__":
    main()
