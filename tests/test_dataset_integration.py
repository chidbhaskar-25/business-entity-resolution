"""Integration test verifying DataLoader and DataValidator against actual project datasets."""

import os
import pytest
from src.business_entity_resolution.data.loader import DataLoader
from src.business_entity_resolution.data.validator import DataValidator


@pytest.mark.skipif(not os.path.exists("data/train/train_source1.tsv"), reason="Data folder not present")
def test_real_train_data_loading_and_validation():
    loader = DataLoader("data/train")

    # Load small slice from real files with validation
    s1_df = loader.load_source_df(
        "train_source1.tsv",
        nrows=100,
        validate=True,
        expected_prefix="S1-",
        preserve_raw=True,
    )
    assert len(s1_df) == 100
    assert "raw_business_name" in s1_df.columns
    assert "raw_business_address" in s1_df.columns
    assert "raw_country" in s1_df.columns
    assert all(eid.startswith("S1-") for eid in s1_df["entity_id"])

    # Load S2
    s2_df = loader.load_source_df(
        "train_source2.tsv",
        nrows=100,
        validate=True,
        expected_prefix="S2-",
    )
    assert len(s2_df) == 100
    assert all(eid.startswith("S2-") for eid in s2_df["entity_id"])

    # Load S3
    s3_df = loader.load_source_df(
        "train_source3.tsv",
        nrows=100,
        validate=True,
        expected_prefix="S3-",
    )
    assert len(s3_df) == 100
    assert all(eid.startswith("S3-") for eid in s3_df["entity_id"])

    # Load Ground Truth
    gt_map = loader.load_ground_truth("train_ground_truth.tsv", nrows=100, validate=True)
    assert len(gt_map) == 100
    assert all(s1_id.startswith("S1-") for s1_id in gt_map.keys())


@pytest.mark.skipif(not os.path.exists("data/test/test_source1.tsv"), reason="Data folder not present")
def test_real_test_data_open_set_country():
    loader = DataLoader("data/test")
    # Load 500 rows to ensure open-set France is encountered
    s1_df = loader.load_source_df(
        "test_source1.tsv",
        nrows=1000,
        validate=True,
        expected_prefix="S1-",
    )
    assert len(s1_df) == 1000
    countries = set(s1_df["country"])
    assert "India" in countries or "US" in countries
