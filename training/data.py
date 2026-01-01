from typing import List, Tuple

from sklearn.model_selection import train_test_split


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
