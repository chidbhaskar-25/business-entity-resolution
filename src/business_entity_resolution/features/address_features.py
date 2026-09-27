"""Pairwise address feature extraction for business entity resolution."""

from typing import Dict, List, Optional, Set
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from ..preprocessing.address_normalizer import AddressNormalizer, NormalizedAddress


class AddressFeatureExtractor:
    """Extracts explainable pairwise similarity features for business addresses."""

    FEATURE_NAMES = [
        "addr_exact_match",
        "addr_levenshtein_sim",
        "addr_token_jaccard",
        "addr_token_overlap_count",
        "addr_token_overlap_ratio",
        "addr_numeric_jaccard",
        "addr_numeric_overlap_count",
        "addr_numeric_exact_match",
        "addr_token_sort_ratio",
        "addr_tfidf_cosine",
        "addr_len_diff_abs",
        "addr_len_diff_ratio",
        "addr_is_missing_either",
        "addr_is_missing_both",
    ]

    def __init__(
        self,
        normalizer: Optional[AddressNormalizer] = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
    ):
        self.normalizer = normalizer or AddressNormalizer()
        self.tfidf_vectorizer = tfidf_vectorizer

    def fit_tfidf(self, address_corpus: List[str]) -> "AddressFeatureExtractor":
        """Fits TF-IDF vectorizer strictly on training reference addresses."""
        clean_corpus = [s for s in address_corpus if s and str(s).strip()]
        min_df = 1 if len(clean_corpus) < 10 else 2
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=min_df,
            max_features=5000,
            dtype=np.float32,
        )
        if clean_corpus:
            try:
                self.tfidf_vectorizer.fit(clean_corpus)
            except ValueError:
                self.tfidf_vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_features=5000,
                    dtype=np.float32,
                )
                self.tfidf_vectorizer.fit(clean_corpus)
        return self

    def extract_features(
        self,
        s1_addr: str,
        target_addr: str,
        norm_s1: Optional[NormalizedAddress] = None,
        norm_target: Optional[NormalizedAddress] = None,
        s1_tfidf_vec: Optional[object] = None,
        target_tfidf_vec: Optional[object] = None,
    ) -> Dict[str, float]:
        """Extracts pairwise address features between Source 1 and Target entity."""
        a1 = norm_s1 or self.normalizer.normalize(s1_addr)
        a2 = norm_target or self.normalizer.normalize(target_addr)

        is_missing_both = 1.0 if a1.is_empty and a2.is_empty else 0.0
        is_missing_either = 1.0 if a1.is_empty or a2.is_empty else 0.0

        if is_missing_either:
            return {
                "addr_exact_match": 1.0 if is_missing_both else 0.0,
                "addr_levenshtein_sim": 0.0,
                "addr_token_jaccard": 0.0,
                "addr_token_overlap_count": 0.0,
                "addr_token_overlap_ratio": 0.0,
                "addr_numeric_jaccard": 0.0,
                "addr_numeric_overlap_count": 0.0,
                "addr_numeric_exact_match": 0.0,
                "addr_token_sort_ratio": 0.0,
                "addr_tfidf_cosine": 0.0,
                "addr_len_diff_abs": float(abs(len(a1.normalized) - len(a2.normalized))),
                "addr_len_diff_ratio": 1.0,
                "addr_is_missing_either": is_missing_either,
                "addr_is_missing_both": is_missing_both,
            }

        s1_norm = a1.normalized
        t_norm = a2.normalized

        # 1. Exact match
        exact_match = 1.0 if s1_norm == t_norm else 0.0

        # 2. Levenshtein edit similarity
        lev_sim = float(Levenshtein.normalized_similarity(s1_norm, t_norm))

        # 3. Token overlap and Jaccard
        tokens1 = set(a1.tokens)
        tokens2 = set(a2.tokens)
        intersection = tokens1 & tokens2
        overlap_count = float(len(intersection))
        union = tokens1 | tokens2
        token_jaccard = (overlap_count / len(union)) if union else 0.0
        min_tokens = min(len(tokens1), len(tokens2))
        overlap_ratio = (overlap_count / min_tokens) if min_tokens > 0 else 0.0

        # 4. Numeric token alignment (house number, postal code, phone)
        nums1 = set(a1.numeric_tokens)
        nums2 = set(a2.numeric_tokens)
        if nums1 and nums2:
            num_intersection = nums1 & nums2
            num_overlap_count = float(len(num_intersection))
            num_jaccard = num_overlap_count / len(nums1 | nums2)
            num_exact_match = 1.0 if nums1 == nums2 else 0.0
        elif not nums1 and not nums2:
            num_overlap_count = 0.0
            num_jaccard = 0.5  # Neutral indicator when neither address has numbers
            num_exact_match = 1.0
        else:
            num_overlap_count = 0.0
            num_jaccard = 0.0
            num_exact_match = 0.0

        # 5. Token sort ratio
        token_sort_ratio = fuzz.token_sort_ratio(s1_norm, t_norm) / 100.0

        # 6. TF-IDF Cosine Similarity
        tfidf_cosine = 0.0
        if s1_tfidf_vec is not None and target_tfidf_vec is not None:
            dot = (s1_tfidf_vec.multiply(target_tfidf_vec)).sum()
            tfidf_cosine = float(min(max(dot, 0.0), 1.0))
        elif self.tfidf_vectorizer is not None:
            try:
                vecs = self.tfidf_vectorizer.transform([s1_norm, t_norm])
                dot = (vecs[0].multiply(vecs[1])).sum()
                tfidf_cosine = float(min(max(dot, 0.0), 1.0))
            except Exception:
                tfidf_cosine = 0.0
        else:
            char_overlap = len(a1.char_ngrams & a2.char_ngrams)
            char_union = len(a1.char_ngrams | a2.char_ngrams)
            tfidf_cosine = (char_overlap / char_union) if char_union else 0.0

        # 7. Length differences
        len1, len2 = len(s1_norm), len(t_norm)
        len_diff_abs = float(abs(len1 - len2))
        len_diff_ratio = len_diff_abs / max(len1, len2, 1)

        return {
            "addr_exact_match": exact_match,
            "addr_levenshtein_sim": round(lev_sim, 5),
            "addr_token_jaccard": round(token_jaccard, 5),
            "addr_token_overlap_count": overlap_count,
            "addr_token_overlap_ratio": round(overlap_ratio, 5),
            "addr_numeric_jaccard": round(num_jaccard, 5),
            "addr_numeric_overlap_count": num_overlap_count,
            "addr_numeric_exact_match": num_exact_match,
            "addr_token_sort_ratio": round(token_sort_ratio, 5),
            "addr_tfidf_cosine": round(tfidf_cosine, 5),
            "addr_len_diff_abs": len_diff_abs,
            "addr_len_diff_ratio": round(len_diff_ratio, 5),
            "addr_is_missing_either": 0.0,
            "addr_is_missing_both": 0.0,
        }
