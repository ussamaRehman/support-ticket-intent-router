import csv
import ssl
from pathlib import Path
from typing import List, Tuple
from urllib.error import URLError
from urllib.request import urlopen

from sklearn.model_selection import train_test_split

TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/"
    "banking_data/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/"
    "banking_data/test.csv"
)
CACHE_DIR = Path("data") / "banking77"


def load_sample_split(
    random_state: int = 42, test_size: float = 0.2
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load a tiny placeholder dataset for wiring the pipeline end-to-end."""
    samples = [
        ("I need help with my invoice", "billing"),
        ("Why was I charged twice this month?", "billing"),
        ("Can I get a refund for last week?", "billing"),
        ("Update my payment method", "billing"),
        ("My app keeps crashing on launch", "technical"),
        ("The website is showing a 500 error", "technical"),
        ("I cannot reset my password", "account"),
        ("Please change the email on my account", "account"),
        ("Two factor authentication is not working", "account"),
        ("My account was locked after failed logins", "account"),
        ("The feature is slow and unresponsive", "technical"),
        ("Billing statement does not match usage", "billing"),
    ]
    texts = [text for text, _ in samples]
    labels = [label for _, label in samples]
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def load_banking77_split(
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    """Load the Banking77 dataset from raw CSVs with a deterministic label map."""
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "train.csv"
    test_path = cache_dir / "test.csv"

    if not train_path.exists():
        _download_csv(TRAIN_URL, train_path)
    if not test_path.exists():
        _download_csv(TEST_URL, test_path)

    X_train, y_train_labels = _load_csv(train_path)
    X_test, y_test_labels = _load_csv(test_path)

    label_names = sorted(set(y_train_labels + y_test_labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y_train = [label_to_id[label] for label in y_train_labels]
    y_test = [label_to_id[label] for label in y_test_labels]

    return X_train, y_train, X_test, y_test, label_names


def _download_csv(url: str, destination: Path) -> None:
    try:
        _download_with_context(url, destination, ssl.create_default_context())
    except URLError as err:
        if isinstance(err.reason, ssl.SSLError):
            try:
                import certifi
            except ImportError as import_err:
                raise RuntimeError(
                    "SSL certificate verification failed. Install certifi or ensure your "
                    "system certificate store is available."
                ) from import_err
            _download_with_context(
                url,
                destination,
                ssl.create_default_context(cafile=certifi.where()),
            )
        else:
            raise


def _download_with_context(url: str, destination: Path, context: ssl.SSLContext) -> None:
    with urlopen(url, context=context) as response, destination.open("wb") as handle:
        handle.write(response.read())


def _load_csv(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(
            handle,
            quotechar="\"",
            delimiter=",",
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
        )
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            text, label = row[0].strip(), row[1].strip()
            if not text or not label:
                continue
            texts.append(text)
            labels.append(label)
    return texts, labels
