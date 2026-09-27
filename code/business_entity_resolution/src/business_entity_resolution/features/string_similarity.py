"""Fast string and token similarity metrics."""

import math
from typing import List, Set, Tuple
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ..preprocessing.tokenizer import TextTokenizer


class StringSimilarity:
    """Computes various text and token similarity scores."""

    @staticmethod
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """Normalized Levenshtein similarity between 0.0 and 1.0."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return Levenshtein.normalized_similarity(s1, s2)

    @staticmethod
    def token_sort_ratio(s1: str, s2: str) -> float:
        """Token sort ratio normalized to [0.0, 1.0]."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return fuzz.token_sort_ratio(s1, s2) / 100.0

    @staticmethod
    def token_set_ratio(s1: str, s2: str) -> float:
        """Token set ratio normalized to [0.0, 1.0]."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return fuzz.token_set_ratio(s1, s2) / 100.0

    @staticmethod
    def jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
        """Jaccard index between two token sets."""
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
        """Character n-gram Jaccard similarity."""
        ngrams1 = TextTokenizer.tokenize_char_ngrams(s1, n=n)
        ngrams2 = TextTokenizer.tokenize_char_ngrams(s2, n=n)
        return StringSimilarity.jaccard_similarity(ngrams1, ngrams2)

    @staticmethod
    def numeric_token_similarity(text1: str, text2: str) -> Tuple[float, int, int]:
        """Calculates Jaccard overlap, shared count, and total count of numbers."""
        nums1 = TextTokenizer.extract_numbers(text1)
        nums2 = TextTokenizer.extract_numbers(text2)

        if not nums1 and not nums2:
            # Both have no numbers -> neutral 0.5 similarity
            return 0.5, 0, 0
        if not nums1 or not nums2:
            # One has numbers, the other doesn't
            return 0.0, 0, len(nums1 | nums2)

        shared = len(nums1 & nums2)
        total = len(nums1 | nums2)
        jaccard = shared / total if total > 0 else 0.0
        return jaccard, shared, total
