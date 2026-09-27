"""Unit tests for pairwise feature generation layer."""

import numpy as np
import pytest

from src.business_entity_resolution.features.address_features import AddressFeatureExtractor
from src.business_entity_resolution.features.country_features import CountryFeatureExtractor
from src.business_entity_resolution.features.feature_builder import FeatureBuilder
from src.business_entity_resolution.features.name_features import NameFeatureExtractor


@pytest.fixture
def name_extractor():
    return NameFeatureExtractor()


@pytest.fixture
def address_extractor():
    return AddressFeatureExtractor()


@pytest.fixture
def country_extractor():
    return CountryFeatureExtractor()


@pytest.fixture
def feature_builder():
    return FeatureBuilder()


def test_name_features_exact_match(name_extractor):
    feats = name_extractor.extract_features(
        s1_name="Maure Williams Colombier Inc",
        target_name="Maure Williams Colombier Inc",
    )
    assert feats["name_exact_match"] == 1.0
    assert feats["name_exact_clean_match"] == 1.0
    assert feats["name_levenshtein_sim"] == 1.0
    assert feats["name_token_jaccard"] == 1.0
    assert feats["name_len_diff_abs"] == 0.0
    assert feats["name_is_missing"] == 0.0


def test_name_features_noisy_pair(name_extractor):
    # Realistic noise: 'Wilblims' vs 'Williams', legal suffix dropped
    feats = name_extractor.extract_features(
        s1_name="Maure Williams Colombier Inc",
        target_name="Maure Wilblims Colombier",
    )
    assert feats["name_exact_match"] == 0.0
    assert feats["name_levenshtein_sim"] >= 0.75
    assert feats["name_token_jaccard"] >= 0.40
    assert feats["name_token_overlap_count"] >= 2.0
    assert feats["name_token_overlap_ratio"] >= 0.60
    assert feats["name_char3_jaccard"] >= 0.50


def test_name_features_missing(name_extractor):
    feats = name_extractor.extract_features(s1_name="", target_name="Some Store")
    assert feats["name_is_missing"] == 1.0
    assert feats["name_levenshtein_sim"] == 0.0


def test_address_features_numeric_overlap(address_extractor):
    # S1 'Af-684' vs S2 'AF-0684' with matching phone '9487203'
    feats = address_extractor.extract_features(
        s1_addr="Af-684, Nandgram, Ph. 989, Ghaziabad 9487203",
        target_addr="AF-0684, NANDGRAM, GHAZIABAD, 9487203",
    )
    assert feats["addr_numeric_exact_match"] == 0.0 or feats["addr_numeric_jaccard"] > 0.5
    assert feats["addr_numeric_overlap_count"] >= 2.0  # Shared '684' and '9487203'
    assert feats["addr_levenshtein_sim"] > 0.70
    assert feats["addr_token_jaccard"] >= 0.50
    assert feats["addr_is_missing_either"] == 0.0


def test_address_features_missing_address(address_extractor):
    feats = address_extractor.extract_features(
        s1_addr="85 Wayne Avenue, Ticonderoga, NY",
        target_addr="",
    )
    assert feats["addr_is_missing_either"] == 1.0
    assert feats["addr_is_missing_both"] == 0.0
    assert feats["addr_levenshtein_sim"] == 0.0


def test_country_features_open_set(country_extractor):
    # Validates open-set matching on France
    feats_fr = country_extractor.extract_features("France", "France")
    assert feats_fr["country_exact_match"] == 1.0
    assert feats_fr["country_is_missing"] == 0.0

    # Cross country mismatch
    feats_mismatch = country_extractor.extract_features("US", "India")
    assert feats_mismatch["country_exact_match"] == 0.0


def test_feature_builder_e2e(feature_builder):
    s1_rec = {
        "entity_id": "S1-001",
        "business_name": "Maure Williams Colombier Inc",
        "business_address": "85 Wayne Avenue, Ticonderoga, NY",
        "country": "US",
    }
    t_match = {
        "entity_id": "S2-001",
        "business_name": "Maure Wilblims Colombier",
        "business_address": "85 Wanye Avenue, Ticonderoga",
        "country": "US",
    }
    t_non_match = {
        "entity_id": "S3-999",
        "business_name": "Different Bakery LLC",
        "business_address": "123 Elm St, Dallas, TX",
        "country": "US",
    }

    feats_pos = feature_builder.extract_pair_dict(s1_rec, t_match)
    feats_neg = feature_builder.extract_pair_dict(s1_rec, t_non_match)

    # Positive match should have high composite heuristic score
    assert feats_pos["composite_heuristic_score"] > 0.75
    # Non match should have low composite score
    assert feats_neg["composite_heuristic_score"] < 0.25

    # Check matrix building
    pairs = [("S1-001", "S2-001"), ("S1-001", "S3-999")]
    s1_lookup = {"S1-001": s1_rec}
    target_lookup = {"S2-001": t_match, "S3-999": t_non_match}

    X, valid_pairs = feature_builder.build_feature_matrix(pairs, s1_lookup, target_lookup)
    assert X.shape == (2, len(feature_builder.feature_names))
    assert valid_pairs == pairs
    assert not np.isnan(X).any()
