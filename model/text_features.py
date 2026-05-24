from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

from model.config import (
    FRAUD_KEYWORD_WEIGHT,
    INJECTION_KEYWORD_WEIGHT,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "your", "you", "our", "this",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KEYWORDS_PATH = DATA_DIR / "keywords.json"


def _load_keywords(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Keyword file must be a JSON object.")

    loaded_keywords: list[set[str]] = []
    for category in ("fraud", "injection"):
        raw_items = payload.get(category)
        if not isinstance(raw_items, list):
            raise ValueError(f"Keyword list missing or invalid: {category}")

        category_terms = {
            str(item).strip().lower()
            for item in raw_items
            if str(item).strip()
        }
        if not category_terms:
            raise ValueError(f"Keyword list is empty: {category}")
        loaded_keywords.append(category_terms)

    return loaded_keywords[0], loaded_keywords[1]


FRAUD_KEYWORDS, INJECTION_KEYWORDS = _load_keywords(KEYWORDS_PATH)


def normalize_text(text: str) -> str:
    return (text or "").lower().strip()


def tokenize_words(text: str) -> List[str]:
    tokens = re.findall(r"[a-z]+", normalize_text(text))
    return [token for token in tokens if token not in STOP_WORDS]


@lru_cache(maxsize=None)
def keyword_pattern(keyword: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in keyword.lower().split() if part]
    if not parts:
        return re.compile(r"$^")
    joined = r"\s+".join(parts)
    return re.compile(rf"\b{joined}\b", flags=re.IGNORECASE)


def find_matches(text: str, keywords: Sequence[str]) -> List[str]:
    normalized = normalize_text(text)
    matches: List[str] = []
    for keyword in sorted(keywords):
        if keyword_pattern(keyword).search(normalized):
            matches.append(keyword)
    return matches


def extract_features(text: str, fraud_matches: Sequence[str], injection_matches: Sequence[str]) -> List[str]:
    words = tokenize_words(text)
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
    return features