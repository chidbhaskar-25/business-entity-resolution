"""Comprehensive API tests for all FastAPI endpoints in app.backend."""

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Tests GET / root documentation endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Business Entity Resolution API" in data["title"]
    assert len(data["endpoints"]) == 9


def test_get_health():
    """Tests GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model" in data
    assert "optimal_threshold" in data
    assert data["train_data_available"] is True
    assert data["test_data_available"] is True
    assert "timestamp" in data


def test_get_dataset_stats():
    """Tests GET /dataset/stats endpoint."""
    response = client.get("/dataset/stats")
    assert response.status_code == 200
    data = response.json()
    assert "train" in data
    assert "test" in data
    assert "train_source1.tsv" in data["train"]
    assert "test_source1.tsv" in data["test"]
    assert "ground_truth" in data


def test_get_metrics():
    """Tests GET /metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "evaluation_metrics" in data
    assert "threshold_metrics" in data


def test_get_experiments():
    """Tests GET /experiments endpoint."""
    response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert data["total"] >= 6
    exp_ids = [e["experiment_id"] for e in data["experiments"]]
    assert "A" in exp_ids
    assert "E" in exp_ids


def test_post_evaluate_endpoint():
    """Tests POST /evaluate endpoint with current best model."""
    payload = {
        "model_path": "models/best_model.joblib",
        "dataset_path": "outputs/train_dataset.npz",
        "threshold": 0.88,
        "optimize_threshold": False,
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "hist_gradient_boosting"
    assert data["threshold_applied"] == 0.88
    assert data["macro_f05"] > 0.90
    assert data["singleton_entities"] > 0


def test_post_train_endpoint():
    """Tests POST /train endpoint with logistic_regression baseline."""
    payload = {
        "model_type": "logistic_regression",
        "dataset_path": "outputs/train_dataset.npz",
        "val_size": 0.20,
        "random_seed": 42,
        "output_model_path": "models/test_api_model.joblib",
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "logistic_regression"
    assert data["train_pairs"] > 0
    assert data["val_pairs"] > 0
    assert "pairwise_metrics" in data
    assert "entity_macro_metrics" in data
    assert data["entity_macro_metrics"]["macro_f05"] > 0.80


def test_post_predict_endpoint():
    """Tests POST /predict endpoint on a small sample of test entities."""
    payload = {
        "sample_size": 20,
        "target_sample_size": 2000,
        "threshold": 0.88,
        "model_path": "models/best_model.joblib",
        "chunk_size": 20,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["sample_size"] == 20
    assert data["validation_passed"] is True
    assert len(data["validation_errors"]) == 0


def test_post_validate_endpoint():
    """Tests POST /validate endpoint."""
    payload = {
        "matching_path": "outputs/matching_results.tsv",
        "candidate_path": "outputs/candidate_pairs.tsv",
        "check_target_ids": False,
        "s1_sample_size": 20,
    }
    response = client.post("/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert "PASS" in data["status"]
    assert data["criteria_checks"]["correct_tsv_format"] is True
    assert data["criteria_checks"]["matches_subset_of_candidates"] is True


def test_get_matches_for_entity():
    """Tests GET /matches/{source1_entity_id} for existing entity and 404 for unknown."""
    # First test with known entity S1-714132312 from outputs/matching_results.tsv
    response = client.get("/matches/S1-714132312")
    assert response.status_code == 200
    data = response.json()
    assert data["source1_entity_id"] == "S1-714132312"
    assert "business_name" in data
    assert "country" in data
    assert isinstance(data["matched_entity_ids"], list)
    assert isinstance(data["candidate_entity_ids"], list)

    # Test unknown entity (404)
    response_404 = client.get("/matches/S1-999999999999")
    assert response_404.status_code == 404

    # Test invalid ID format (400)
    response_400 = client.get("/matches/INVALID_ID")
    assert response_400.status_code == 400
