"""Naive Bayes classifier for email security analysis."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from model.config import (
    LAPLACE_SMOOTHING,
    POSTERIOR_TEMPERATURE,
    TITLE_WEIGHT,
)
from model.text_features import (
    FRAUD_KEYWORDS,
    INJECTION_KEYWORDS,
    extract_features,
    find_matches,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "final_dataset.csv"
# model cache
_MODEL = None

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
    """Build a Naive Bayes token-frequency model for each class/category."""
    word_counts: Dict[str, Counter] = defaultdict(Counter)
    class_counts = Counter()

    for text, label in rows:
        class_counts[label] += 1
        fraud_matches = find_matches(text, FRAUD_KEYWORDS)
        injection_matches = find_matches(text, INJECTION_KEYWORDS)
        tokens = extract_features(text, fraud_matches, injection_matches)
        for token in tokens:
            word_counts[label][token] += 1

    labels = sorted(class_counts.keys())

    vocabulary = set()
    token_totals: Dict[str, int] = {}
    for label in labels:
        vocabulary.update(word_counts[label].keys())
        token_totals[label] = sum(word_counts[label].values())

    return {
        "labels": labels,
        "word_counts": dict(word_counts),
        "class_counts": dict(class_counts),
        "vocabulary": vocabulary,
        "token_totals": token_totals,
    }

def _get_model() -> Dict[str, object]:
    """Load and cache the model, training it on first use."""
    global _MODEL
    if _MODEL is None:
        rows = _load_dataset()
        _MODEL = _build_model(rows)
    return _MODEL


def predict_email(text: str = "", title: str = "", body: str = "") -> Dict[str, Any]:
    """Classify an email using Naive Bayes with keyword-based features."""
    model = _get_model()

    raw_title = (title or "").strip()
    raw_body = (body or text or "").strip()
    combined_text = "\n".join(part for part in [raw_title, raw_body] if part)

    # Weight title signals more strongly by repeating it.
    weighted_title = " ".join([raw_title] * TITLE_WEIGHT).strip()
    weighted_text = f"{weighted_title} {raw_body}".strip()

    fraud_matches = find_matches(combined_text, FRAUD_KEYWORDS)
    injection_matches = find_matches(combined_text, INJECTION_KEYWORDS)
    tokens = extract_features(weighted_text, fraud_matches, injection_matches)

    labels = model["labels"]
    word_counts = model["word_counts"]
    class_counts = model["class_counts"]
    token_totals = model["token_totals"]
    vocabulary = model["vocabulary"]
    vocab_size = max(1, len(vocabulary))
    total_docs = sum(class_counts.values()) or 1

    scores: Dict[str, float] = {}

    for label in labels:
        class_count = class_counts.get(label, 1)
        label_word_counts = word_counts.get(label, {})
        total_words = token_totals.get(label, 0)

        # Empirical class prior for Naive Bayes behavior.
        log_prob = math.log(class_count / total_docs)

        for token in tokens:
            word_freq = label_word_counts.get(token, 0)
            prob = (word_freq + LAPLACE_SMOOTHING) / (total_words + LAPLACE_SMOOTHING * vocab_size)
            log_prob += math.log(prob)

        scores[label] = log_prob

    if not scores:
        raise ValueError("No labels available in model. Check the dataset.")

    predicted_label = max(scores, key=scores.get)

    token_count = max(1, len(tokens))
    calibrated_logits = {label: (scores[label] / token_count) for label in labels}

    max_score = max(calibrated_logits.values())
    exp_scores = {
        label: math.exp((calibrated_logits[label] - max_score) / POSTERIOR_TEMPERATURE)
        for label in labels
    }
    total_exp = sum(exp_scores.values()) or 1.0
    normalized = {label: exp_scores[label] / total_exp for label in labels}

    is_malicious = predicted_label in {"fraud", "injection"}

    return {
        "is_fraud": normalized.get("fraud", 0.0),
        "is_injection": normalized.get("injection", 0.0),
        "is_safe": normalized.get("safe", 0.0),
        "predicted_label": predicted_label,
        "status": "MALICIOUS" if is_malicious else "SAFE",
        "score": round(scores.get(predicted_label, 0.0), 3),
        "raw_scores": {label: round(score, 6) for label, score in scores.items()},
        "fraud_matches": fraud_matches,
        "injection_matches": injection_matches,
    }