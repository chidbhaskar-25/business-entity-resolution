"""Unified feature builder integrating name, address, and country features."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .address_features import AddressFeatureExtractor
from .country_features import CountryFeatureExtractor
from .name_features import NameFeatureExtractor
from ..preprocessing.address_normalizer import AddressNormalizer
from ..preprocessing.name_normalizer import BusinessNameNormalizer
from ..utils.logger import setup_logger

logger = setup_logger("features.builder")


class FeatureBuilder:
    """Extracts a comprehensive, interpretable pairwise feature vector for candidate pairs."""

    def __init__(
        self,
        name_extractor: Optional[NameFeatureExtractor] = None,
        address_extractor: Optional[AddressFeatureExtractor] = None,
        country_extractor: Optional[CountryFeatureExtractor] = None,
    ):
        self.name_extractor = name_extractor or NameFeatureExtractor()
        self.address_extractor = address_extractor or AddressFeatureExtractor()
        self.country_extractor = country_extractor or CountryFeatureExtractor()

        # Cache of normalized records for fast batch extraction
        self._name_normalizer = self.name_extractor.normalizer
        self._addr_normalizer = self.address_extractor.normalizer

        # Dynamically build feature names list
        self.feature_names = self._init_feature_names()

    def _init_feature_names(self) -> List[str]:
        names = []
        names.extend(self.name_extractor.FEATURE_NAMES)
        names.extend(self.address_extractor.FEATURE_NAMES)
        names.extend(self.country_extractor.FEATURE_NAMES)
        names.extend([
            "target_is_source2",
            "name_length_s1",
            "name_length_target",
            "addr_length_s1",
            "addr_length_target",
            "name_token_count_diff",
            "composite_heuristic_score",
        ])
        return names

    def fit_reference_tfidf(
        self,
        s1_names: List[str],
        s1_addresses: List[str],
    ) -> "FeatureBuilder":
        """Fits TF-IDF vectorizers on reference training data to prevent data leakage."""
        logger.info(f"Fitting TF-IDF on {len(s1_names):,} training reference entities...")
        self.name_extractor.fit_tfidf(s1_names)
        self.address_extractor.fit_tfidf(s1_addresses)
        return self

    def extract_pair_dict(
        self,
        s1_record: Dict[str, str],
        target_record: Dict[str, str],
        norm_n1: Optional[NormalizedName] = None,
        norm_n2: Optional[NormalizedName] = None,
        norm_a1: Optional[NormalizedAddress] = None,
        norm_a2: Optional[NormalizedAddress] = None,
        s1_name_vec: Optional[object] = None,
        target_name_vec: Optional[object] = None,
        s1_addr_vec: Optional[object] = None,
        target_addr_vec: Optional[object] = None,
    ) -> Dict[str, float]:
        """Extracts complete feature dictionary for a single candidate pair."""
        s1_name = str(s1_record.get("business_name", ""))
        s1_addr = str(s1_record.get("business_address", ""))
        s1_country = str(s1_record.get("country", ""))

        t_id = str(target_record.get("entity_id", ""))
        t_name = str(target_record.get("business_name", ""))
        t_addr = str(target_record.get("business_address", ""))
        t_country = str(target_record.get("country", ""))

        # 1. Preprocess / Normalize
        n1 = norm_n1 or self._name_normalizer.normalize(s1_name)
        n2 = norm_n2 or self._name_normalizer.normalize(t_name)
        a1 = norm_a1 or self._addr_normalizer.normalize(s1_addr)
        a2 = norm_a2 or self._addr_normalizer.normalize(t_addr)

        # 2. Extract sub-features
        name_feats = self.name_extractor.extract_features(
            s1_name,
            t_name,
            norm_s1=n1,
            norm_target=n2,
            s1_tfidf_vec=s1_name_vec,
            target_tfidf_vec=target_name_vec,
        )
        addr_feats = self.address_extractor.extract_features(
            s1_addr,
            t_addr,
            norm_s1=a1,
            norm_target=a2,
            s1_tfidf_vec=s1_addr_vec,
            target_tfidf_vec=target_addr_vec,
        )
        cntry_feats = self.country_extractor.extract_features(s1_country, t_country)

        # 3. Structural and source indicators
        target_is_s2 = 1.0 if t_id.startswith("S2-") else 0.0
        name_len_s1 = float(len(n1.normalized))
        name_len_target = float(len(n2.normalized))
        addr_len_s1 = float(len(a1.normalized))
        addr_len_target = float(len(a2.normalized))
        token_count_diff = float(abs(len(n1.tokens) - len(n2.tokens)))

        # 4. Composite heuristic score
        if a1.is_empty or a2.is_empty:
            composite = (
                0.70 * name_feats["name_token_sort_ratio"]
                + 0.30 * name_feats["name_levenshtein_sim"]
            )
        else:
            composite = (
                0.35 * name_feats["name_token_sort_ratio"]
                + 0.15 * name_feats["name_levenshtein_sim"]
                + 0.25 * addr_feats["addr_token_sort_ratio"]
                + 0.25 * addr_feats["addr_numeric_jaccard"]
            )
        composite = min(max(composite, 0.0), 1.0) * cntry_feats["country_exact_match"]

        combined = {}
        combined.update(name_feats)
        combined.update(addr_feats)
        combined.update(cntry_feats)
        combined.update({
            "target_is_source2": target_is_s2,
            "name_length_s1": name_len_s1,
            "name_length_target": name_len_target,
            "addr_length_s1": addr_len_s1,
            "addr_length_target": addr_len_target,
            "name_token_count_diff": token_count_diff,
            "composite_heuristic_score": round(composite, 5),
        })
        return combined

    def build_feature_matrix(
        self,
        candidate_pairs: List[Tuple[str, str]],
        s1_lookup: Dict[str, Dict[str, str]],
        target_lookup: Dict[str, Dict[str, str]],
    ) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """Extracts feature matrix for a list of (s1_id, target_id) candidate pairs with caching.

        Args:
            candidate_pairs: List of (s1_id, target_id) tuples.
            s1_lookup: Mapping of s1_id -> record dict.
            target_lookup: Mapping of target_id -> record dict.

        Returns:
            Tuple of (X: np.ndarray of shape (N, num_features), valid_pairs: List of pairs).
        """
        if not candidate_pairs:
            return np.empty((0, len(self.feature_names)), dtype=np.float32), []

        # 1. Identify unique entity IDs
        unique_s1_ids = list({p[0] for p in candidate_pairs if p[0] in s1_lookup})
        unique_t_ids = list({p[1] for p in candidate_pairs if p[1] in target_lookup})

        # 2. Pre-normalize unique records
        s1_n = {sid: self._name_normalizer.normalize(str(s1_lookup[sid].get("business_name", ""))) for sid in unique_s1_ids}
        s1_a = {sid: self._addr_normalizer.normalize(str(s1_lookup[sid].get("business_address", ""))) for sid in unique_s1_ids}
        t_n = {tid: self._name_normalizer.normalize(str(target_lookup[tid].get("business_name", ""))) for tid in unique_t_ids}
        t_a = {tid: self._addr_normalizer.normalize(str(target_lookup[tid].get("business_address", ""))) for tid in unique_t_ids}

        # 3. Batch transform TF-IDF
        s1_name_vecs: Dict[str, object] = {}
        t_name_vecs: Dict[str, object] = {}
        s1_addr_vecs: Dict[str, object] = {}
        t_addr_vecs: Dict[str, object] = {}

        if self.name_extractor.tfidf_vectorizer is not None:
            if unique_s1_ids:
                m1 = self.name_extractor.tfidf_vectorizer.transform([s1_n[sid].normalized for sid in unique_s1_ids])
                for i, sid in enumerate(unique_s1_ids):
                    s1_name_vecs[sid] = m1[i]
            if unique_t_ids:
                m2 = self.name_extractor.tfidf_vectorizer.transform([t_n[tid].normalized for tid in unique_t_ids])
                for i, tid in enumerate(unique_t_ids):
                    t_name_vecs[tid] = m2[i]

        if self.address_extractor.tfidf_vectorizer is not None:
            if unique_s1_ids:
                m1 = self.address_extractor.tfidf_vectorizer.transform([s1_a[sid].normalized for sid in unique_s1_ids])
                for i, sid in enumerate(unique_s1_ids):
                    s1_addr_vecs[sid] = m1[i]
            if unique_t_ids:
                m2 = self.address_extractor.tfidf_vectorizer.transform([t_a[tid].normalized for tid in unique_t_ids])
                for i, tid in enumerate(unique_t_ids):
                    t_addr_vecs[tid] = m2[i]

        # 4. Extract rows
        rows: List[List[float]] = []
        valid_pairs: List[Tuple[str, str]] = []

        for s1_id, target_id in candidate_pairs:
            r1 = s1_lookup.get(s1_id)
            r2 = target_lookup.get(target_id)
            if not r1 or not r2:
                continue

            feats = self.extract_pair_dict(
                r1,
                r2,
                norm_n1=s1_n.get(s1_id),
                norm_n2=t_n.get(target_id),
                norm_a1=s1_a.get(s1_id),
                norm_a2=t_a.get(target_id),
                s1_name_vec=s1_name_vecs.get(s1_id),
                target_name_vec=t_name_vecs.get(target_id),
                s1_addr_vec=s1_addr_vecs.get(s1_id),
                target_addr_vec=t_addr_vecs.get(target_id),
            )
            row_vector = [feats.get(name, 0.0) for name in self.feature_names]
            rows.append(row_vector)
            valid_pairs.append((s1_id, target_id))

        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32), []

        X = np.array(rows, dtype=np.float32)
        return X, valid_pairs
