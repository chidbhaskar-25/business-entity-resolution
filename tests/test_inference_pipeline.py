"""Unit tests for inference pipeline and submission TSV generation."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.business_entity_resolution.data.validator import OutputValidator
from src.business_entity_resolution.inference.pipeline import InferencePipeline
from src.business_entity_resolution.models.matcher import BaselineMatcher


def test_inference_pipeline_mock_execution(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create mock training reference S1 file for TF-IDF fitting
    train_s1 = pd.DataFrame([
        {"entity_id": "S1-T1", "business_name": "Acme Tools Inc", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-T2", "business_name": "Global Tech LLC", "business_address": "456 Tech Park", "country": "India"},
    ])
    train_s1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)

    # Create mock test S1 file (3 entities: 1 single match, 1 multi match, 1 singleton)
    test_s1 = pd.DataFrame([
        {"entity_id": "S1-101", "business_name": "Walmart Supercenter", "business_address": "100 Store Rd, Bentonville", "country": "US"},
        {"entity_id": "S1-102", "business_name": "Target Express", "business_address": "200 Target Way, Minneapolis", "country": "US"},
        {"entity_id": "S1-103", "business_name": "Unique Local Bakery", "business_address": "999 Nowhere Lane", "country": "France"},
    ])
    test_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

    # Create mock test S2 file
    test_s2 = pd.DataFrame([
        {"entity_id": "S2-201", "business_name": "Walmart Supercenter Store", "business_address": "100 Store Rd", "country": "US"},
        {"entity_id": "S2-202", "business_name": "Target Express Mart", "business_address": "200 Target Way", "country": "US"},
    ])
    test_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

    # Create mock test S3 file
    test_s3 = pd.DataFrame([
        {"entity_id": "S3-301", "business_name": "Walmart Bentonville", "business_address": "100 Store Road", "country": "US"},
        {"entity_id": "S3-302", "business_name": "Other Unrelated Corp", "business_address": "555 Avenue", "country": "France"},
    ])
    test_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    # Train a simple model
    model_path = str(tmp_path / "model.joblib")
    th_path = str(tmp_path / "threshold.json")
    X_train = np.random.randn(20, 36).astype(np.float32)
    y_train = np.array([1 if i % 2 == 0 else 0 for i in range(20)], dtype=np.int32)
    matcher = BaselineMatcher(model_type="logistic_regression", random_state=42)
    matcher.fit(X_train, y_train)
    matcher.save(model_path)

    # Setup pipeline
    pipeline = InferencePipeline(
        model=matcher,
        threshold=0.50,
        max_candidates_per_entity=10,
    )

    matching_out = str(tmp_path / "matching_results.tsv")
    candidate_out = str(tmp_path / "candidate_pairs.tsv")

    report = pipeline.run_full_inference(
        test_dir=str(test_dir),
        train_dir=str(train_dir),
        matching_output_path=matching_out,
        candidate_output_path=candidate_out,
        chunk_size=2,
    )

    assert report.is_valid is True
    assert os.path.exists(matching_out)
    assert os.path.exists(candidate_out)

    # Inspect matching_results.tsv
    with open(matching_out, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 4  # 1 header + 3 entities
    assert lines[0] == "source1_entity_id\tmatched_entity_ids\n"

    # Inspect candidate_pairs.tsv
    with open(candidate_out, "r", encoding="utf-8") as f:
        cand_lines = f.readlines()
    assert len(cand_lines) == 4
    assert cand_lines[0] == "source1_entity_id\tcandidate_entity_ids\n"

    # Run submission validator
    val_report = OutputValidator.validate_submission(
        matching_path=matching_out,
        candidate_path=candidate_out,
        test_dir=str(test_dir),
    )
    assert val_report.is_valid is True
    assert len(val_report.errors) == 0


def test_inference_singleton_preservation_and_thresholding(tmp_path):
    """Verifies that predicted singletons preserve empty match lists and thresholding works."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    train_s1 = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Entity One", "business_address": "Addr 1", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Entity Two", "business_address": "Addr 2", "country": "US"},
    ])
    train_s1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)

    test_s1 = pd.DataFrame([
        {"entity_id": "S1-A", "business_name": "Alpha Corp", "business_address": "100 Alpha St", "country": "US"},
        {"entity_id": "S1-B", "business_name": "Beta Corp", "business_address": "200 Beta St", "country": "US"},
    ])
    test_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

    test_s2 = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Alpha Corp", "business_address": "100 Alpha St", "country": "US"},
    ])
    test_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

    test_s3 = pd.DataFrame([
        {"entity_id": "S3-1", "business_name": "Alpha Corporation", "business_address": "100 Alpha Street", "country": "US"},
    ])
    test_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    # Use a dummy matcher where predict_proba returns 0.0 (all singletons) with threshold=0.5
    class DummySingletonMatcher:
        model_type = "dummy_singleton"
        def predict_proba(self, X):
            return np.zeros(len(X), dtype=np.float32)

    pipeline = InferencePipeline(
        model=DummySingletonMatcher(),
        threshold=0.50,
        max_candidates_per_entity=5,
    )

    matching_out = str(tmp_path / "matching_results.tsv")
    candidate_out = str(tmp_path / "candidate_pairs.tsv")

    report = pipeline.run_full_inference(
        test_dir=str(test_dir),
        train_dir=str(train_dir),
        matching_output_path=matching_out,
        candidate_output_path=candidate_out,
    )

    assert report.is_valid is True

    # Both entities must be empty singletons in matching_results.tsv
    with open(matching_out, "r", encoding="utf-8") as f:
        lines = [l.strip("\n") for l in f.readlines()]

    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-A\t"
    assert lines[2] == "S1-B\t"

    # Now test with dummy matcher returning 1.0 (all matches above threshold)
    class DummyMultiMatcher:
        model_type = "dummy_multi"
        def predict_proba(self, X):
            return np.ones(len(X), dtype=np.float32)

    pipeline_multi = InferencePipeline(
        model=DummyMultiMatcher(),
        threshold=0.50,
        max_candidates_per_entity=5,
    )

    report_multi = pipeline_multi.run_full_inference(
        test_dir=str(test_dir),
        train_dir=str(train_dir),
        matching_output_path=matching_out,
        candidate_output_path=candidate_out,
    )
    assert report_multi.is_valid is True

    with open(matching_out, "r", encoding="utf-8") as f:
        m_lines = [l.strip("\n") for l in f.readlines()]

    # S1-A matched both S2-1 and S3-1 (multiple matches)
    parts = m_lines[1].split("\t")
    assert parts[0] == "S1-A"
    matched_ids = parts[1].split(",")
    assert "S2-1" in matched_ids
    assert "S3-1" in matched_ids
    assert len(matched_ids) == len(set(matched_ids))  # no duplicates


def test_inference_pipeline_loads_saved_artifacts():
    """Verifies that InferencePipeline successfully loads models/best_model.joblib and threshold."""
    pipeline = InferencePipeline(
        model_path="models/best_model.joblib",
        threshold_path="models/best_threshold.json",
    )
    assert pipeline.model is not None
    assert pipeline.model.is_fitted is True
    assert pipeline.threshold == 0.88

