"""Unit tests for DataLoader."""

import os
import tempfile
import pandas as pd
import pytest

from src.business_entity_resolution.data.loader import DataLoader


@pytest.fixture
def temp_data_dir():
    """Creates a temporary directory with mock TSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock train_source1.tsv (including open-set countries)
        s1_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-001\tAcme Corp\t123 Main St\tUS\n"
            "S1-002\tTata Consultancy\tGateway Tower\tIndia\n"
            "S1-003\tBoulangerie Paris\t15 Rue de Rivoli\tFrance\n"
            "S1-004\tEmpty Addr Corp\t\tUS\n"
        )
        with open(os.path.join(tmpdir, "train_source1.tsv"), "w", encoding="utf-8") as f:
            f.write(s1_content)

        # Create mock train_ground_truth.tsv
        gt_content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-001\tS2-101,S3-201\n"
            "S1-002\tS2-102\n"
            "S1-003\t\n"
            "S1-004\tS3-204\n"
        )
        with open(os.path.join(tmpdir, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
            f.write(gt_content)

        yield tmpdir


def test_loader_reads_tab_separated(temp_data_dir):
    loader = DataLoader(temp_data_dir)
    df = loader.load_source_df("train_source1.tsv")
    assert len(df) == 4
    assert list(df.columns[:4]) == ["entity_id", "business_name", "business_address", "country"]
    assert df.loc[0, "entity_id"] == "S1-001"
    assert df.loc[0, "business_name"] == "Acme Corp"


def test_loader_preserves_raw_data(temp_data_dir):
    loader = DataLoader(temp_data_dir)
    df = loader.load_source_df("train_source1.tsv", preserve_raw=True)
    assert "raw_business_name" in df.columns
    assert "raw_business_address" in df.columns
    assert "raw_country" in df.columns
    assert df.loc[0, "raw_business_name"] == "Acme Corp"
    assert df.loc[3, "business_address"] == ""
    assert df.loc[3, "raw_business_address"] == ""


def test_loader_open_set_countries(temp_data_dir):
    loader = DataLoader(temp_data_dir)
    df = loader.load_source_df("train_source1.tsv")
    countries = set(df["country"])
    assert "US" in countries
    assert "India" in countries
    assert "France" in countries  # Open set verification


def test_loader_streaming_chunks(temp_data_dir):
    loader = DataLoader(temp_data_dir)
    chunks = list(loader.stream_source_chunks("train_source1.tsv", chunk_size=2))
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert chunks[0].iloc[0]["entity_id"] == "S1-001"
    assert chunks[1].iloc[0]["entity_id"] == "S1-003"


def test_loader_ground_truth(temp_data_dir):
    loader = DataLoader(temp_data_dir)
    gt_map = loader.load_ground_truth("train_ground_truth.tsv")
    assert len(gt_map) == 4
    assert gt_map["S1-001"] == {"S2-101", "S3-201"}
    assert gt_map["S1-002"] == {"S2-102"}
    assert gt_map["S1-003"] == set()  # Singleton
    assert gt_map["S1-004"] == {"S3-204"}


def test_loader_file_not_found():
    loader = DataLoader("non_existent_directory")
    with pytest.raises(FileNotFoundError):
        loader.load_source_df("missing.tsv")
