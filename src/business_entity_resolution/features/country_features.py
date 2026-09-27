"""Pairwise country feature extraction for open-set entity resolution."""

from typing import Dict, List
from ..preprocessing.text_utils import strip_accents


class CountryFeatureExtractor:
    """Extracts open-set country equality and missing indicator features."""

    FEATURE_NAMES = [
        "country_exact_match",
        "country_is_missing",
    ]

    @staticmethod
    def normalize_country(country: str) -> str:
        """Normalizes open-set country string without restricting to closed list."""
        if not country or not str(country).strip():
            return ""
        return strip_accents(str(country)).strip().upper()

    def extract_features(
        self,
        s1_country: str,
        target_country: str,
    ) -> Dict[str, float]:
        """Extracts country similarity features between Source 1 and Target entity."""
        c1 = self.normalize_country(s1_country)
        c2 = self.normalize_country(target_country)

        is_missing = 1.0 if not c1 or not c2 else 0.0
        exact_match = 1.0 if c1 and c1 == c2 else 0.0

        return {
            "country_exact_match": exact_match,
            "country_is_missing": is_missing,
        }
