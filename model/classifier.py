"""Simple Naive Bayes classifier for email security analysis.

A beginner-friendly implementation that:
1. Loads training data from CSV
2. Learns word frequencies for each email class (safe/fraud/injection)
3. Classifies new emails by counting matching words

"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Small, explicit stop-word set to keep preprocessing understandable.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "your", "you", "our", "this",
    "please",
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dummy_dataset.csv"

# Global model cache
_MODEL = None


def preprocess_text(text: str) -> str:
    """Normalize input text before Naive Bayes scoring.
    
    Steps:
    1) Lowercase text
    2) Tokenize using alphabetic word boundaries
    3) Remove English stop words
    4) Re-join into a clean string for token counting
    """
    normalized = (text or "").lower()
    tokens = re.findall(r"[a-z]+", normalized)
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(filtered_tokens)


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into word tokens."""
    if not text:
        return []
    return text.split()


def _load_dataset() -> List[Tuple[str, str]]:
    """Load labeled examples from CSV as (text, label) tuples."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    rows: List[Tuple[str, str]] = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("Dataset must contain `text` and `label` columns.")

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if text and label:
                rows.append((text, label))

    if not rows:
        raise ValueError("Dataset is empty. Provide at least one labeled row.")

    return rows


def _build_model(rows: List[Tuple[str, str]]) -> Dict[str, object]:
    """Build a simple word frequency model for each class.
    
    For each class (safe/fraud/injection), count how many times each word appears.
    That's it - no smoothing, no complex math.
    """
    # word_counts[label] = Counter of word frequencies for that label
    word_counts: Dict[str, Counter] = defaultdict(Counter)
    class_counts = Counter()  # How many examples per class
    
    for text, label in rows:
        class_counts[label] += 1
        tokens = tokenize(preprocess_text(text))
        for token in tokens:
            word_counts[label][token] += 1
    
    labels = sorted(class_counts.keys())
    
    return {
        "labels": labels,
        "word_counts": dict(word_counts),  # {label: {word: frequency}}
        "class_counts": dict(class_counts),  # {label: example_count}
    }

def _get_model() -> Dict[str, object]:
    """Load and cache the model, training it on first use."""
    global _MODEL
    if _MODEL is None:
        rows = _load_dataset()
        _MODEL = _build_model(rows)
    return _MODEL


def predict_email(text: str) -> Dict[str, Any]:
    """Classify an email by counting word matches in each class.
    
    Simple approach:
    1. Extract and preprocess words from the email
    2. For each class, count how many of those words appeared in training
    3. Return the class with the highest count
    """
    model = _get_model()
    cleaned_text = preprocess_text(text)
    tokens = tokenize(cleaned_text)
    
    labels = model["labels"]
    word_counts = model["word_counts"]
    class_counts = model["class_counts"]
    
    # For each class, count matching words
    scores: Dict[str, int] = {}
    for label in labels:
        match_count = 0
        for token in tokens:
            # Count how many times this word appears in this class's training data
            match_count += word_counts.get(label, {}).get(token, 0)
        scores[label] = match_count
    
    # Predicted class is the one with highest word match count
    predicted_label = max(scores, key=scores.get)
    predicted_score = scores[predicted_label]
    
    # Calculate simple percentages for display
    total_matches = sum(scores.values()) or 1
    safe_score = scores.get("safe", 0) / total_matches
    fraud_score = scores.get("fraud", 0) / total_matches
    injection_score = scores.get("injection", 0) / total_matches
    
    # Mark as malicious if fraud or injection is predicted
    is_malicious = predicted_label in {"fraud", "injection"}
    
    return {
        "is_fraud": round(fraud_score, 4),
        "is_injection": round(injection_score, 4),
        "is_safe": round(safe_score, 4),
        "predicted_label": predicted_label,
        "status": "MALICIOUS" if is_malicious else "SAFE",
        "match_count": predicted_score,
    }
