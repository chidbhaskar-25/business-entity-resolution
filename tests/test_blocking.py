"""Unit tests for multi-strategy candidate blocking layer."""

import os
import tempfile
import pandas as pd
import pytest

from src.business_entity_resolution.blocking.address_blocking import AddressBlocker
from src.business_entity_resolution.blocking.blocker import CompositeBlocker
from src.business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from src.business_entity_resolution.blocking.name_blocking import NameBlocker


def test_name_blocker_keys():
    blocker = NameBlocker(prefix_len=6)
    keys = blocker.generate_keys("Maure Williams Colombier Inc")
    # Exact key
    assert "name_exact:maurewilliamscolombier" in keys
    # Prefix key
    assert "name_pfx:maurew" in keys
    # Token keys
    assert "name_tok:williams" in keys
    assert "name_tok:colombier" in keys
    # Bigram key
    assert "name_bi:maure_williams" in keys


def test_address_blocker_keys():
    blocker = AddressBlocker()
    keys = blocker.generate_keys("Af-684, Nandgram, Ghaziabad 9487203")
    # Number keys
    assert "addr_num:684" in keys
    assert "addr_num:9487203" in keys
    # Locality key
    assert "addr_tok:nandgram" in keys
    assert "addr_tok:ghaziabad" in keys


def test_address_blocker_empty():
    blocker = AddressBlocker()
    keys = blocker.generate_keys("")
    assert len(keys) == 0


def test_composite_blocker_country_isolation():
    blocker = CompositeBlocker()

    # Index US target
    blocker.index_target_record("S2-US01", "Acme Corp", "123 Main St", "US")
    # Index France target with identical name
    blocker.index_target_record("S2-FR01", "Acme Corp", "15 Rue Paris", "France")

    # Query in US
    us_cands = blocker.retrieve_candidates("Acme Corp", "123 Main St", "US")
    assert "S2-US01" in us_cands
    assert "S2-FR01" not in us_cands  # Must not cross countries

    # Query in France
    fr_cands = blocker.retrieve_candidates("Acme Corp", "15 Rue Paris", "France")
    assert "S2-FR01" in fr_cands
    assert "S2-US01" not in fr_cands


def test_candidate_generator_recall_and_reduction():
    # Setup mock targets
    targets_df = pd.DataFrame({
        "entity_id": ["S2-001", "S3-002", "S2-003", "S3-004"],
        "business_name": [
            "Maure Wilblims Colombier",
            "Dahlia Ponr Reliable Scientific",
            "Random Other Store",
            "Dréxkor",
        ],
        "business_address": [
            "",
            "630 45ND TERRACE, KANSAS CITY",
            "999 Elm St",
            "85 Wanye Avenue, Ticonderoga Townshiip",
        ],
        "country": ["US", "US", "US", "US"],
    })

    # Setup mock S1
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": [
            "Maure Williams Colombier Inc",
            "Dahlia Power Reliable Scientific LLC",
        ],
        "business_address": [
            "85 Wayne Avenue, Ticonderoga, NY",
            "630 45th Terrace, Kansas City, MO",
        ],
        "country": ["US", "US"],
    })

    # Ground truth mapping
    gt = {
        "S1-001": {"S2-001", "S3-004"},
        "S1-002": {"S3-002"},
    }

    generator = CandidateGenerator(max_candidates_per_entity=10)
    generator.fit_target_records(targets_df)

    cands_map, metrics = generator.generate_candidates_batch(s1_df, ground_truth=gt)

    # Verify S1-001 retrieved S2-001 (via name token/bigram) and S3-004 (via address street/num)
    assert "S2-001" in cands_map["S1-001"]
    assert "S3-004" in cands_map["S1-001"]

    # Verify S1-002 retrieved S3-002 (via name and address tokens)
    assert "S3-002" in cands_map["S1-002"]

    # Candidate recall must be 1.0 (all 3 true links found!)
    assert metrics.candidate_recall == 1.0
    assert metrics.true_pairs_found == 3
    assert metrics.total_true_pairs == 3
    assert metrics.reduction_ratio > 0.0

    # Test saving metrics
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_metrics.json")
        generator.save_metrics(metrics, json_path)
        assert os.path.exists(json_path)
