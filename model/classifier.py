"""Simple Naive Bayes classifier for email security analysis.

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


# for fraud/injection detection.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "your", "you", "our", "this",
}

# Domain-specific keywords for fraud detection (phishing, urgency, deception)
FRAUD_KEYWORDS = {
    "verify", "confirm", "urgent", "immediate", "action", "required",
    "click", "link", "update", "password", "account", "suspended",
    "alert", "security", "unusual", "activity", "re-activate", "confirm identity",
    "bank", "paypal", "amazon", "apple", "microsoft", "verify account",
    "act now", "limited time", "expire", "claim", "reward", "congratulations",
}

# Domain-specific keywords for prompt injection detection
INJECTION_KEYWORDS = {
    "select", "insert", "delete", "drop", "exec", "execute", "script",
    "eval", "import", "function", "lambda", "print", "return", "class",
    "def", "query", "database", "table", "sql", "code", "system",
    "command", "bash", "shell", "python", "java", "javascript",
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
    4) Add special tokens for detected fraud/injection keywords (so model learns their importance)
    5) Re-join into a clean string for token counting
    """
    normalized = (text or "").lower()
    tokens = re.findall(r"[a-z]+", normalized)
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS]
    
    # Add domain-specific keyword markers - so the model learns they're important, not post-processing
    fraud_found = any(kw in normalized for kw in FRAUD_KEYWORDS)
    injection_found = any(kw in normalized for kw in INJECTION_KEYWORDS)
    
    if fraud_found:
        filtered_tokens.append("__fraud_keyword_detected__")
    if injection_found:
        filtered_tokens.append("__injection_keyword_detected__")
    
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


def _calculate_heuristic_scores(text: str) -> Dict[str, float]:
    """Calculate simple heuristic indicators for fraud and injection.
    
    Returns:
        Dictionary with fraud_boost and injection_boost scores (0.0 to 1.0)
    """
    fraud_boost = 0.0
    injection_boost = 0.0
    
    lower_text = text.lower()
    
    # 1. URL detection - phishing indicator
    if re.search(r'http[s]?://|www\.', lower_text):
        fraud_boost += 0.15
    
    # 2. Count fraud keywords in original text
    fraud_keyword_count = sum(1 for keyword in FRAUD_KEYWORDS if keyword in lower_text)
    if fraud_keyword_count > 0:
        fraud_boost += min(0.25, fraud_keyword_count * 0.05)
    
    # 3. Excessive special characters - injection indicator
    special_char_count = len(re.findall(r'[!@#$%^&*()=+\[\]{};:\'",.<>?/\\]', text))
    if len(text) > 10:
        special_char_ratio = special_char_count / len(text)
        if special_char_ratio > 0.15:  # High density of special chars
            injection_boost += min(0.20, special_char_ratio * 0.5)
    
    # 4. Code/SQL patterns - injection indicator
    code_patterns = [
        r'\bselect\b.*\bfrom\b',  # SQL SELECT
        r'\bdrop\b.*\btable\b',   # SQL DROP
        r'<script|javascript:|eval|exec',  # XSS/Code patterns
    ]
    for pattern in code_patterns:
        if re.search(pattern, lower_text, re.IGNORECASE):
            injection_boost += 0.20
            break
    
    # 5. Count injection keywords
    injection_keyword_count = sum(1 for keyword in INJECTION_KEYWORDS if keyword in lower_text)
    if injection_keyword_count > 1:  # Multiple code-related words
        injection_boost += min(0.25, injection_keyword_count * 0.08)
    
    # 6. Urgent/scarcity language - fraud indicator
    urgent_patterns = ['urgent', 'immediate', 'expire', 'limited time', 'act now', 'click here']
    urgent_count = sum(1 for phrase in urgent_patterns if phrase in lower_text)
    if urgent_count > 0:
        fraud_boost += min(0.15, urgent_count * 0.05)
    
    return {
        "fraud_boost": min(fraud_boost, 1.0),
        "injection_boost": min(injection_boost, 1.0)
    }


def predict_email(text: str) -> Dict[str, Any]:
    """Classify an email using Naïve Bayes with integrated domain keywords.
    
    Approach:
    1. Preprocess text (includes domain keywords as special tokens)
    2. Model learns from training data which keywords indicate fraud/injection
    3. Calculate log probabilities using learned patterns
    4. Return normalized probabilities for each category
    """
    import math
    
    model = _get_model()
    cleaned_text = preprocess_text(text)
    tokens = tokenize(cleaned_text)
    
    labels = model["labels"]
    word_counts = model["word_counts"]
    class_counts = model["class_counts"]
    
    # Calculate total unique words across all classes (vocabulary size)
    vocab = set()
    for label in labels:
        vocab.update(word_counts.get(label, {}).keys())
    vocab_size = len(vocab)
    
    # For each class, calculate log probability with Laplace smoothing
    scores: Dict[str, float] = {}
    for label in labels:
        class_count = class_counts.get(label, 1)
        label_word_counts = word_counts.get(label, {})
        total_words = sum(label_word_counts.values()) or 1
        
        # Prior probability of the class
        log_prob = math.log(class_count / (sum(class_counts.values()) or 1))
        
        # For each token, add log P(token | class) with Laplace smoothing
        for token in tokens:
            word_freq = label_word_counts.get(token, 0)
            # Laplace smoothing: (count + 1) / (total_words + vocab_size)
            prob = (word_freq + 1) / (total_words + vocab_size)
            log_prob += math.log(prob)
        
        scores[label] = log_prob
    
    # Predicted class is the one with highest score
    predicted_label = max(scores, key=scores.get)
    
    # Convert scores to normalized probabilities for display
    # Shift scores to prevent numeric underflow, then exponentiate and normalize
    max_score = max(scores.values())
    exp_scores = {label: math.exp(scores[label] - max_score) for label in labels}
    total_exp = sum(exp_scores.values())
    normalized = {label: exp_scores[label] / total_exp for label in labels}
    
    # Mark as malicious if fraud or injection is predicted
    is_malicious = predicted_label in {"fraud", "injection"}
    
    return {
        "is_fraud": round(normalized.get("fraud", 0), 4),
        "is_injection": round(normalized.get("injection", 0), 4),
        "is_safe": round(normalized.get("safe", 0), 4),
        "predicted_label": predicted_label,
        "status": "MALICIOUS" if is_malicious else "SAFE",
        "match_count": round(scores[predicted_label], 2),
    }
