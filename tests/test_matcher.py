"""Unit tests for ML matcher, training pipeline, and predictor."""

import os
import tempfile
import numpy as np
import pytest

from src.business_entity_resolution.data.label_builder import LabelBuilder
from src.business_entity_resolution.models.matcher import BaselineMatcher
from src.business_entity_resolution.models.predict_model import EntityPredictor
from src.business_entity_resolution.models.train_model import train_baseline


def test_baseline_matcher_fit_predict_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")

        # Synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        # Positive label correlated with feature 0
        y = (X[:, 0] > 0.0).astype(np.int32)
        feature_names = [f"f_{i}" for i in range(10)]

        matcher = BaselineMatcher(model_type="logistic_regression", random_state=42)
        matcher.fit(X, y, feature_names=feature_names)

        probs = matcher.predict_proba(X)
        assert probs.shape == (100,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        preds = matcher.predict(X, threshold=0.5)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

        # Save and reload with joblib
        matcher.save(model_path)
        assert os.path.exists(model_path)

        loaded_matcher = BaselineMatcher.load(model_path)
        loaded_probs = loaded_matcher.predict_proba(X)
        np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)


def test_baseline_matcher_gbm():
    np.random.seed(42)
    X = np.random.randn(80, 5).astype(np.float32)
    y = (X[:, 1] + X[:, 2] > 0).astype(np.int32)

    matcher = BaselineMatcher(model_type="hist_gradient_boosting", random_state=42)
    matcher.fit(X, y)

    probs = matcher.predict_proba(X)
    assert probs.shape == (80,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_train_baseline_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "test_dataset.npz")
        model_path = os.path.join(tmpdir, "baseline_model.joblib")
        metrics_path = os.path.join(tmpdir, "metrics.json")

        # Create mini synthetic dataset with 20 S1 entities, each having 5 candidates
        np.random.seed(42)
        pairs = []
        for i in range(20):
            s1_id = f"S1-{i:03d}"
            for j in range(5):
                t_id = f"S2-{i * 5 + j:04d}"
                pairs.append((s1_id, t_id))

        N = len(pairs)
        X = np.random.randn(N, 6).astype(np.float32)
        # Give every entity 1 positive candidate (j=0) and 4 negatives (j>0)
        y = np.array([1 if j == 0 else 0 for i in range(20) for j in range(5)], dtype=np.int32)

        feat_names = [f"feat_{k}" for k in range(6)]
        LabelBuilder.save_dataset_npz(dataset_path, X, y, pairs, feat_names)

        # Run training
        metrics = train_baseline(
            dataset_path=dataset_path,
            output_model_path=model_path,
            metrics_output_path=metrics_path,
            model_type="logistic_regression",
            val_size=0.25,
            random_seed=42,
        )

        assert os.path.exists(model_path)
        assert os.path.exists(metrics_path)
        assert "pairwise_metrics" in metrics
        assert "competition_macro_metrics_at_0_50" in metrics
        assert metrics["pairwise_metrics"]["roc_auc"] >= 0.0


def test_entity_predictor():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_predictor_model.joblib")

        np.random.seed(42)
        X = np.random.randn(50, 4).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(np.int32)
        y[0] = 1
        y[1] = 0

        matcher = BaselineMatcher(model_type="logistic_regression", random_state=42)
        matcher.fit(X, y, feature_names=[f"col_{i}" for i in range(4)])
        matcher.save(model_path)

        predictor = EntityPredictor(model_path)
        pairs = [(f"S1-{i // 5}", f"S2-{i}") for i in range(50)]
        matches_map = predictor.predict_matches_map(pairs, X, threshold=0.50, all_s1_ids=[f"S1-{k}" for k in range(12)])

        assert len(matches_map) >= 10
        # Check singleton S1-11 is in map with empty set
        assert "S1-11" in matches_map
        assert isinstance(matches_map["S1-11"], set)
