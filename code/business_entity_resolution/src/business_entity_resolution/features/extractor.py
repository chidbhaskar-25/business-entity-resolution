"""Pairwise feature extraction for candidate matches."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .string_similarity import StringSimilarity
from ..preprocessing.normalizer import TextNormalizer
from ..preprocessing.tokenizer import TextTokenizer
from ..utils.logger import setup_logger

logger = setup_logger("features.extractor")

FEATURE_NAMES = [
    "name_exact_match",
    "name_exact_stripped_match",
    "name_levenshtein",
    "name_token_sort",
    "name_token_set",
    "name_word_jaccard",
    "name_char3_jaccard",
    "name_len_diff_ratio",
    "addr_exact_match",
    "addr_token_sort",
    "addr_word_jaccard",
    "addr_numeric_jaccard",
    "addr_numeric_shared_count",
    "addr_is_empty",
    "addr_len_diff_ratio",
    "country_exact_match",
    "target_is_s2",
    "composite_heuristic_score",
]


class FeatureExtractor:
    """Extracts explainable, high-signal pairwise features for business record pairs."""

    def __init__(self):
        self.feature_names = FEATURE_NAMES

    def extract_pair_features(
        self,
        s1_name: str,
        s1_addr: str,
        s1_country: str,
        target_id: str,
        target_name: str,
        target_addr: str,
        target_country: str,
    ) -> List[float]:
        """Calculates feature vector for a single pair of records."""
        # 1. Normalizations
        norm_s1_name = TextNormalizer.normalize_name(s1_name, strip_legal=False)
        norm_t_name = TextNormalizer.normalize_name(target_name, strip_legal=False)

        clean_s1_name = TextNormalizer.normalize_name(s1_name, strip_legal=True)
        clean_t_name = TextNormalizer.normalize_name(target_name, strip_legal=True)

        norm_s1_addr = TextNormalizer.normalize_address(s1_addr)
        norm_t_addr = TextNormalizer.normalize_address(target_addr)

        norm_s1_country = TextNormalizer.normalize_country(s1_country)
        norm_t_country = TextNormalizer.normalize_country(target_country)

        # 2. Name Features
        name_exact = 1.0 if norm_s1_name and norm_s1_name == norm_t_name else 0.0
        name_exact_stripped = 1.0 if clean_s1_name and clean_s1_name == clean_t_name else 0.0
        name_lev = StringSimilarity.levenshtein_ratio(norm_s1_name, norm_t_name)
        name_tok_sort = StringSimilarity.token_sort_ratio(norm_s1_name, norm_t_name)
        name_tok_set = StringSimilarity.token_set_ratio(norm_s1_name, norm_t_name)

        tokens_s1_name = set(TextTokenizer.tokenize_words(norm_s1_name))
        tokens_t_name = set(TextTokenizer.tokenize_words(norm_t_name))
        name_jaccard = StringSimilarity.jaccard_similarity(tokens_s1_name, tokens_t_name)
        name_char3 = StringSimilarity.char_ngram_jaccard(norm_s1_name, norm_t_name, n=3)

        len1, len2 = len(norm_s1_name), len(norm_t_name)
        name_len_diff = abs(len1 - len2) / max(len1, len2, 1)

        # 3. Address Features
        addr_empty = 1.0 if not norm_s1_addr or not norm_t_addr else 0.0
        if addr_empty:
            addr_exact = 0.0
            addr_tok_sort = 0.0
            addr_jaccard = 0.0
            addr_num_jaccard = 0.0
            addr_num_shared = 0.0
            addr_len_diff = 1.0
        else:
            addr_exact = 1.0 if norm_s1_addr == norm_t_addr else 0.0
            addr_tok_sort = StringSimilarity.token_sort_ratio(norm_s1_addr, norm_t_addr)
            tokens_s1_addr = set(TextTokenizer.tokenize_words(norm_s1_addr))
            tokens_t_addr = set(TextTokenizer.tokenize_words(norm_t_addr))
            addr_jaccard = StringSimilarity.jaccard_similarity(tokens_s1_addr, tokens_t_addr)

            num_jacc, num_shared, _ = StringSimilarity.numeric_token_similarity(
                norm_s1_addr, norm_t_addr
            )
            addr_num_jaccard = num_jacc
            addr_num_shared = float(min(num_shared, 5)) / 5.0

            alen1, alen2 = len(norm_s1_addr), len(norm_t_addr)
            addr_len_diff = abs(alen1 - alen2) / max(alen1, alen2, 1)

        # 4. Country & Source Features
        country_match = 1.0 if norm_s1_country and norm_s1_country == norm_t_country else 0.0
        target_is_s2 = 1.0 if target_id.startswith("S2-") else 0.0

        # 5. Composite Heuristic Score
        # If address is available, combine name and address; else rely on name
        if addr_empty:
            composite = 0.8 * name_tok_sort + 0.2 * name_lev
        else:
            composite = (
                0.40 * name_tok_sort
                + 0.20 * name_lev
                + 0.25 * addr_tok_sort
                + 0.15 * addr_num_jaccard
            )
        composite = min(max(composite, 0.0), 1.0) * country_match

        return [
            name_exact,
            name_exact_stripped,
            name_lev,
            name_tok_sort,
            name_tok_set,
            name_word_jaccard,
            name_char3,
            name_len_diff,
            addr_exact,
            addr_tok_sort,
            addr_jaccard,
            addr_num_jaccard,
            addr_num_shared,
            addr_empty,
            addr_len_diff,
            country_match,
            target_is_s2,
            composite,
        ]

    def extract_features_for_candidates(
        self,
        s1_records: Dict[str, Dict[str, str]],
        target_records: Dict[str, Dict[str, str]],
        candidate_map: Dict[str, List[str]],
    ) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """Extracts features for all candidate pairs in candidate_map.

        Args:
            s1_records: Mapping of s1_id -> record dict.
            target_records: Mapping of target_id -> record dict.
            candidate_map: Mapping of s1_id -> list of candidate target_ids.

        Returns:
            Tuple of (feature_matrix X as np.ndarray, list of (s1_id, target_id) pairs).
        """
        pair_list: List[Tuple[str, str]] = []
        feature_rows: List[List[float]] = []

        for s1_id, candidates in candidate_map.items():
            r1 = s1_records.get(s1_id)
            if not r1:
                continue
            s1_name = r1.get("business_name", "")
            s1_addr = r1.get("business_address", "")
            s1_country = r1.get("country", "")

            for tid in candidates:
                r2 = target_records.get(tid)
                if not r2:
                    continue
                pair_list.append((s1_id, tid))
                feats = self.extract_pair_features(
                    s1_name=s1_name,
                    s1_addr=s1_addr,
                    s1_country=s1_country,
                    target_id=tid,
                    target_name=r2.get("business_name", ""),
                    target_addr=r2.get("business_address", ""),
                    target_country=r2.get("country", ""),
                )
                feature_rows.append(feats)

        if not feature_rows:
            return np.empty((0, len(self.feature_names))), []

        X = np.array(feature_rows, dtype=np.float32)
        return X, pair_list
