"""Pairwise name feature extraction for business entity resolution."""

from typing import Dict, List, Optional, Set
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from ..preprocessing.name_normalizer import BusinessNameNormalizer, NormalizedName
from ..preprocessing.text_utils import tokenize_char_ngrams, tokenize_words


class NameFeatureExtractor:
    """Extracts explainable pairwise similarity features for business names."""

    FEATURE_NAMES = [
        "name_exact_match",
        "name_exact_clean_match",
        "name_levenshtein_sim",
        "name_token_jaccard",
        "name_token_overlap_count",
        "name_token_overlap_ratio",
        "name_char3_jaccard",
        "name_token_sort_ratio",
        "name_tfidf_cosine",
        "name_len_diff_abs",
        "name_len_diff_ratio",
        "name_same_legal_suffix",
        "name_is_missing",
    ]

    def __init__(
        self,
        normalizer: Optional[BusinessNameNormalizer] = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
    ):
        self.normalizer = normalizer or BusinessNameNormalizer()
        self.tfidf_vectorizer = tfidf_vectorizer

    def fit_tfidf(self, name_corpus: List[str]) -> "NameFeatureExtractor":
        """Fits TF-IDF vectorizer strictly on training reference names to prevent data leakage."""
        clean_corpus = [s for s in name_corpus if s and str(s).strip()]
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
        s1_name: str,
        target_name: str,
        norm_s1: Optional[NormalizedName] = None,
        norm_target: Optional[NormalizedName] = None,
        s1_tfidf_vec: Optional[object] = None,
        target_tfidf_vec: Optional[object] = None,
    ) -> Dict[str, float]:
        """Extracts pairwise name features between Source 1 and Target entity."""
        n1 = norm_s1 or self.normalizer.normalize(s1_name)
        n2 = norm_target or self.normalizer.normalize(target_name)

        s1_norm = n1.normalized
        t_norm = n2.normalized

        # Missing indicator
        is_missing = 1.0 if not s1_norm or not t_norm else 0.0
        if is_missing:
            return {
                "name_exact_match": 0.0,
                "name_exact_clean_match": 0.0,
                "name_levenshtein_sim": 0.0,
                "name_token_jaccard": 0.0,
                "name_token_overlap_count": 0.0,
                "name_token_overlap_ratio": 0.0,
                "name_char3_jaccard": 0.0,
                "name_token_sort_ratio": 0.0,
                "name_tfidf_cosine": 0.0,
                "name_len_diff_abs": float(abs(len(s1_norm) - len(t_norm))),
                "name_len_diff_ratio": 1.0,
                "name_same_legal_suffix": 0.0,
                "name_is_missing": 1.0,
            }

        # 1. Exact match features
        exact_match = 1.0 if s1_norm == t_norm else 0.0
        clean1 = n1.clean_without_legal
        clean2 = n2.clean_without_legal
        exact_clean_match = 1.0 if clean1 and clean1 == clean2 else 0.0

        # 2. Levenshtein edit similarity
        lev_sim = float(Levenshtein.normalized_similarity(s1_norm, t_norm))

        # 3. Token overlap and Jaccard
        tokens1 = set(n1.tokens)
        tokens2 = set(n2.tokens)
        intersection = tokens1 & tokens2
        overlap_count = float(len(intersection))
        union = tokens1 | tokens2
        token_jaccard = (overlap_count / len(union)) if union else 0.0
        min_tokens = min(len(tokens1), len(tokens2))
        overlap_ratio = (overlap_count / min_tokens) if min_tokens > 0 else 0.0

        # 4. Character similarity
        char_jaccard = (
            len(n1.char_ngrams & n2.char_ngrams) / len(n1.char_ngrams | n2.char_ngrams)
            if (n1.char_ngrams | n2.char_ngrams) else 0.0
        )
        token_sort_ratio = fuzz.token_sort_ratio(s1_norm, t_norm) / 100.0

        # 5. TF-IDF Cosine Similarity
        tfidf_cosine = 0.0
        if s1_tfidf_vec is not None and target_tfidf_vec is not None:
            dot = (s1_tfidf_vec.multiply(target_tfidf_vec)).sum()
            tfidf_cosine = float(min(max(dot, 0.0), 1.0))
        elif self.tfidf_vectorizer is not None:
            try:
                vecs = self.tfidf_vectorizer.transform([s1_norm, t_norm])
                # Dot product of normalized vectors equals cosine similarity
                dot = (vecs[0].multiply(vecs[1])).sum()
                tfidf_cosine = float(min(max(dot, 0.0), 1.0))
            except Exception:
                tfidf_cosine = 0.0
        else:
            # Fallback to character n-gram cosine approximation
            tfidf_cosine = char_jaccard

        # 6. Length differences
        len1, len2 = len(s1_norm), len(t_norm)
        len_diff_abs = float(abs(len1 - len2))
        len_diff_ratio = len_diff_abs / max(len1, len2, 1)

        # 7. Legal suffix alignment
        same_legal = (
            1.0
            if n1.detected_legal_suffix and n1.detected_legal_suffix == n2.detected_legal_suffix
            else 0.0
        )

        return {
            "name_exact_match": exact_match,
            "name_exact_clean_match": exact_clean_match,
            "name_levenshtein_sim": round(lev_sim, 5),
            "name_token_jaccard": round(token_jaccard, 5),
            "name_token_overlap_count": overlap_count,
            "name_token_overlap_ratio": round(overlap_ratio, 5),
            "name_char3_jaccard": round(char_jaccard, 5),
            "name_token_sort_ratio": round(token_sort_ratio, 5),
            "name_tfidf_cosine": round(tfidf_cosine, 5),
            "name_len_diff_abs": len_diff_abs,
            "name_len_diff_ratio": round(len_diff_ratio, 5),
            "name_same_legal_suffix": same_legal,
            "name_is_missing": 0.0,
        }
