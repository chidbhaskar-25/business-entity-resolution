"""Unit tests for DataValidator and OutputValidator."""

import os
import tempfile
import pandas as pd
import pytest

from src.business_entity_resolution.data.validator import DataValidator, OutputValidator


def test_validate_source_dataframe_valid():
    df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "business_name": ["Alpha Inc", "Beta LLC", "Gamma Corp"],
        "business_address": ["123 Main St", "456 Oak Rd", "789 Pine Ave"],
        "country": ["US", "India", "France"],  # Open-set countries
    })
    is_valid, errors, warnings = DataValidator.validate_source_dataframe(df, expected_prefix="S1-")
    assert is_valid is True
    assert len(errors) == 0


def test_validate_source_missing_columns():
    df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Alpha Inc"],
    })
    is_valid, errors, warnings = DataValidator.validate_source_dataframe(df, expected_prefix="S1-")
    assert is_valid is False
    assert any("Missing required columns" in e for e in errors)


def test_validate_source_invalid_prefix():
    df = pd.DataFrame({
        "entity_id": ["S2-001", "S1-002"],  # S2-001 in S1 source
        "business_name": ["Alpha Inc", "Beta LLC"],
        "business_address": ["123 Main St", "456 Oak Rd"],
        "country": ["US", "US"],
    })
    is_valid, errors, warnings = DataValidator.validate_source_dataframe(df, expected_prefix="S1-")
    assert is_valid is False
    assert any("without required prefix 'S1-'" in e for e in errors)


def test_validate_source_duplicate_ids():
    df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-001"],
        "business_name": ["Alpha Inc", "Beta LLC"],
        "business_address": ["123 Main St", "456 Oak Rd"],
        "country": ["US", "India"],
    })
    is_valid, errors, warnings = DataValidator.validate_source_dataframe(df, expected_prefix="S1-")
    assert is_valid is False
    assert any("duplicate entity IDs" in e for e in errors)


def test_validate_source_missing_values_safe_handling():
    # Empty address is common in S2/S3; it should trigger a warning, not an error
    df = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002"],
        "business_name": ["Alpha", "Beta"],
        "business_address": ["", None],
        "country": ["France", "India"],
    })
    is_valid, errors, warnings = DataValidator.validate_source_dataframe(df, expected_prefix="S2-")
    assert is_valid is True
    assert len(errors) == 0
    assert any("empty business_address" in w for w in warnings)


def test_validate_source_open_set_country():
    # Supports any open-set country string
    df = pd.DataFrame({
        "entity_id": ["S3-001", "S3-002", "S3-003"],
        "business_name": ["Boulangerie", "Bäckerei", "Izakaya"],
        "business_address": ["Paris", "Berlin", "Tokyo"],
        "country": ["France", "Germany", "Japan"],
    })
    is_valid, errors, warnings = DataValidator.validate_source_dataframe(df, expected_prefix="S3-")
    assert is_valid is True
    assert len(errors) == 0


def test_validate_ground_truth_valid():
    df = pd.DataFrame({
        "source1_entity_id": ["S1-001", "S1-002", "S1-003"],
        "matched_entity_ids": ["S2-101,S3-201", "S3-202", ""],
    })
    is_valid, errors, warnings = DataValidator.validate_ground_truth_dataframe(df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_ground_truth_self_match():
    # S1 self match must be detected and rejected
    df = pd.DataFrame({
        "source1_entity_id": ["S1-001"],
        "matched_entity_ids": ["S1-999,S2-101"],
    })
    is_valid, errors, warnings = DataValidator.validate_ground_truth_dataframe(df)
    assert is_valid is False
    assert any("self-matches" in e for e in errors)


def test_validate_ground_truth_intra_dupes():
    # Repeated IDs within same row must be rejected
    df = pd.DataFrame({
        "source1_entity_id": ["S1-001"],
        "matched_entity_ids": ["S2-101,S2-101"],
    })
    is_valid, errors, warnings = DataValidator.validate_ground_truth_dataframe(df)
    assert is_valid is False
    assert any("duplicate targets within matched_entity_ids" in e for e in errors)


def test_output_validator_tsv_comma_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "matching_results.tsv")
        # Write comma separated instead of tab
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id,matched_entity_ids\nS1-001,S2-101\n")

        _, errors, _ = OutputValidator.validate_id_list_file(
            filepath=csv_file,
            expected_header=["source1_entity_id", "matched_entity_ids"],
            col_label="matched_entity_ids",
        )
        assert any("Header has no TAB but contains commas" in e for e in errors)
