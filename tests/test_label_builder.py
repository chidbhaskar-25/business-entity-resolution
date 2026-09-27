"""Unit tests for ground-truth label builder and dataset serialization."""

import os
import tempfile
import numpy as np
import pytest

from src.business_entity_resolution.data.label_builder import LabelBuilder, LabelValidationReport


def test_build_labels_single_and_multi_match():
    # S1-001 has 2 true matches (multi-match)
    # S1-002 has 1 true match
    gt_map = {
        "S1-001": {"S2-101", "S3-201"},
        "S1-002": {"S2-102"},
    }

    candidate_map = {
        "S1-001": ["S2-101", "S3-201", "S2-999"],  # 2 matches, 1 negative
        "S1-002": ["S2-102", "S3-888"],            # 1 match, 1 negative
    }

    labels, pairs = LabelBuilder.build_labels_for_candidates(candidate_map, gt_map)

    expected_pairs = [
        ("S1-001", "S2-101"),
        ("S1-001", "S3-201"),
        ("S1-001", "S2-999"),
        ("S1-002", "S2-102"),
        ("S1-002", "S3-888"),
    ]
    expected_labels = [1, 1, 0, 1, 0]

    assert pairs == expected_pairs
    assert labels == expected_labels

    # Run validation
    report = LabelBuilder.validate_labels(pairs, labels, gt_map)
    assert report.is_valid is True
    assert report.total_pairs == 5
    assert report.positive_labels == 3
    assert report.negative_labels == 2
    assert report.positive_ratio == 0.6


def test_build_labels_singleton():
    # S1-003 is a singleton (no matches in ground truth)
    gt_map = {
        "S1-003": set(),
    }

    # Blocker generated 3 plausible candidates
    candidate_map = {
        "S1-003": ["S2-301", "S3-302", "S2-303"],
    }

    labels, pairs = LabelBuilder.build_labels_for_candidates(candidate_map, gt_map)

    # All candidates for a singleton MUST strictly receive label 0
    assert labels == [0, 0, 0]
    assert len(pairs) == 3

    report = LabelBuilder.validate_labels(pairs, labels, gt_map)
    assert report.is_valid is True
    assert report.positive_labels == 0
    assert report.negative_labels == 3
    assert report.singleton_pairs_count == 3
    assert report.singleton_positive_count == 0


def test_validate_labels_catches_false_positive():
    gt_map = {
        "S1-001": {"S2-101"},
    }
    pairs = [("S1-001", "S2-999")]  # S2-999 is NOT in ground truth
    labels = [1]                    # Corrupted label

    report = LabelBuilder.validate_labels(pairs, labels, gt_map)
    assert report.is_valid is False
    assert any("False positive label error" in e for e in report.errors)


def test_validate_labels_catches_singleton_positive_violation():
    gt_map = {
        "S1-002": set(),  # Singleton
    }
    pairs = [("S1-002", "S2-500")]
    labels = [1]  # Erroneous positive label on singleton

    report = LabelBuilder.validate_labels(pairs, labels, gt_map)
    assert report.is_valid is False
    assert report.singleton_positive_count == 1
    assert any("CRITICAL: Singleton" in e for e in report.errors)


def test_save_and_load_dataset_npz():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "train_features.npz")

        X_orig = np.random.rand(10, 5).astype(np.float32)
        y_orig = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.int32)
        pairs_orig = [(f"S1-{i:03d}", f"S2-{i:03d}") for i in range(10)]
        feature_names_orig = [f"feat_{i}" for i in range(5)]

        LabelBuilder.save_dataset_npz(
            filepath=npz_path,
            X=X_orig,
            y=y_orig,
            pairs=pairs_orig,
            feature_names=feature_names_orig,
        )
        assert os.path.exists(npz_path)

        # Reload
        X_loaded, y_loaded, pairs_loaded, feat_names_loaded = LabelBuilder.load_dataset_npz(npz_path)

        np.testing.assert_array_almost_equal(X_orig, X_loaded)
        np.testing.assert_array_equal(y_orig, y_loaded)
        assert pairs_orig == pairs_loaded
        assert feature_names_orig == feat_names_loaded
