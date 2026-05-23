"""Naive Bayes classifier for email security analysis.
This module implements a simple Naive Bayes classifier.
Using token frequencies and keyword matches.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "your", "you", "our", "this",
}

# Tunable scoring.
# Adjust confidence behavior.
TITLE_WEIGHT = 2
LAPLACE_SMOOTHING = 1.0
POSTERIOR_TEMPERATURE = 0.5
FRAUD_KEYWORD_WEIGHT = 2
INJECTION_KEYWORD_WEIGHT = 2
FRAUD_SENTENCE_WEIGHT = 3
INJECTION_SENTENCE_WEIGHT = 3

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "final_dataset.csv"
FRAUD_KEYWORDS_PATH = DATA_DIR / "fraud_keywords.txt"
INJECTION_KEYWORDS_PATH = DATA_DIR / "injection_keywords.txt"


def _load_keyword_file(path: Path) -> set[str]:
    """Load newline-delimited keywords from a file in data/. """
    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found: {path}")

    loaded: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip().lower()
            if not line or line.startswith("#"):
                continue
            loaded.add(line)

    if not loaded:
        raise ValueError(f"Keyword file is empty: {path}")

    return loaded


FRAUD_KEYWORDS = _load_keyword_file(FRAUD_KEYWORDS_PATH)
INJECTION_KEYWORDS = _load_keyword_file(INJECTION_KEYWORDS_PATH)

# model cache
_MODEL = None


def _normalize_text(text: str) -> str:
    """Normalize input text for consistent feature extraction."""
    return (text or "").lower().strip()


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphabetic words with stop-word filtering."""
    tokens = re.findall(r"[a-z]+", _normalize_text(text))
    return [token for token in tokens if token not in STOP_WORDS]


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Build a single or multi-word keywords."""
    parts = [re.escape(part) for part in keyword.lower().split() if part]
    if not parts:
        return re.compile(r"$^")
    joined = r"\s+".join(parts)
    return re.compile(rf"\b{joined}\b", flags=re.IGNORECASE)


def _find_matches(text: str, keywords: Sequence[str]) -> List[str]:
    """Return unique keyword/phrase matches that appear in the text."""
    normalized = _normalize_text(text)
    matches: List[str] = []
    for keyword in sorted(keywords):
        pattern = _keyword_pattern(keyword)
        if pattern.search(normalized):
            matches.append(keyword)
    return matches


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-like chunks for explainability"""
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [segment.strip() for segment in candidates if segment and segment.strip()]


def _matched_sentences(text: str, matches: Sequence[str]) -> List[str]:
    """Collect sentence fragments that contain any of the matched terms."""
    if not matches:
        return []

    patterns = [_keyword_pattern(term) for term in matches]
    flagged: List[str] = []
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if any(pattern.search(lowered) for pattern in patterns):
            flagged.append(sentence)
    return flagged


def _sentence_features(text: str, fraud_matches: Sequence[str], injection_matches: Sequence[str]) -> List[str]:
    """Create sentence-level features so malicious clauses inside safe messages stay visible."""
    features: List[str] = []
    fraud_patterns = [_keyword_pattern(term) for term in fraud_matches]
    injection_patterns = [_keyword_pattern(term) for term in injection_matches]

    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        fraud_hits = sum(1 for pattern in fraud_patterns if pattern.search(lowered))
        injection_hits = sum(1 for pattern in injection_patterns if pattern.search(lowered))

        if fraud_hits:
            features.extend(["__fraud_sentence__"] * (FRAUD_SENTENCE_WEIGHT * fraud_hits))
        if injection_hits:
            features.extend(["__injection_sentence__"] * (INJECTION_SENTENCE_WEIGHT * injection_hits))

    return features


def _extract_features(text: str, fraud_matches: Sequence[str], injection_matches: Sequence[str]) -> List[str]:
    """Build features for Naive Bayes scoring."""
    words = _tokenize_words(text)
    if not words:
        return []

    features: List[str] = list(words)
    features.extend(f"{words[index]}_{words[index + 1]}" for index in range(len(words) - 1))

    for _ in fraud_matches:
        features.extend(["__fraud_kw__"] * FRAUD_KEYWORD_WEIGHT)
        features.extend(["__malicious_signal__"] * FRAUD_KEYWORD_WEIGHT)
    for _ in injection_matches:
        features.extend(["__injection_kw__"] * INJECTION_KEYWORD_WEIGHT)
        features.extend(["__malicious_signal__"] * INJECTION_KEYWORD_WEIGHT)

    features.extend(_sentence_features(text, fraud_matches, injection_matches))

    return features


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
    """Build a Naive Bayes token-frequency model for each class."""
    word_counts: Dict[str, Counter] = defaultdict(Counter)
    class_counts = Counter()

    for text, label in rows:
        class_counts[label] += 1
        fraud_matches = _find_matches(text, FRAUD_KEYWORDS)
        injection_matches = _find_matches(text, INJECTION_KEYWORDS)
        tokens = _extract_features(text, fraud_matches, injection_matches)
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
    """Classify an email using Naive Bayes with keyword-based features.

    Args:
        text: Backward-compatible input for single-body callers.
        title: Email subject/title.
        body: Email body.
    """
    model = _get_model()

    raw_title = (title or "").strip()
    raw_body = (body or text or "").strip()
    combined_text = "\n".join(part for part in [raw_title, raw_body] if part)

    # Weight title signals more strongly by repeating it.
    weighted_title = " ".join([raw_title] * TITLE_WEIGHT).strip()
    weighted_text = f"{weighted_title} {raw_body}".strip()

    fraud_matches = _find_matches(combined_text, FRAUD_KEYWORDS)
    injection_matches = _find_matches(combined_text, INJECTION_KEYWORDS)
    tokens = _extract_features(weighted_text, fraud_matches, injection_matches)

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

    flagged_sentences = _matched_sentences(combined_text, [*fraud_matches, *injection_matches])

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
        "flagged_sentences": flagged_sentences,
        "signal_counts": {
            "fraud": len(fraud_matches),
            "injection": len(injection_matches),
        },
    }
