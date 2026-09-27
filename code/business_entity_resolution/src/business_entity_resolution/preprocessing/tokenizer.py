"""Tokenization and token extraction utilities."""

import re
from typing import List, Set


class TextTokenizer:
    """Provides word, n-gram, and numeric tokenization."""

    @staticmethod
    def tokenize_words(text: str, min_len: int = 1) -> List[str]:
        """Extracts alphabetic and alphanumeric tokens."""
        if not text:
            return []
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if min_len > 1:
            tokens = [t for t in tokens if len(t) >= min_len]
        return tokens

    @staticmethod
    def tokenize_char_ngrams(text: str, n: int = 3) -> Set[str]:
        """Extracts character n-grams from text."""
        cleaned = re.sub(r"\s+", " ", text.strip().lower())
        if not cleaned:
            return set()
        if len(cleaned) < n:
            return {cleaned}
        return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}

    @staticmethod
    def extract_numbers(text: str) -> Set[str]:
        """Extracts numeric strings, normalizing leading zeroes (e.g. '0684' -> '684')."""
        if not text:
            return set()
        raw_numbers = re.findall(r"\b\d+\b", text)
        normalized = set()
        for num in raw_numbers:
            val = num.lstrip("0") or "0"
            normalized.add(val)
        return normalized

    @staticmethod
    def extract_address_keys(text: str) -> Set[str]:
        """Extracts candidate blocking keys from normalized address (pincodes, street numbers)."""
        if not text:
            return set()
        # Numbers (including normalized without leading zeroes)
        numbers = TextTokenizer.extract_numbers(text)
        return {f"num_{n}" for n in numbers}
