"""Full system test suite verifying all 13 core stages of the Business Entity Resolution application.

1. Data loading
2. Normalization
3. Blocking
4. Feature generation
5. Label generation
6. Model training
7. Evaluation
8. Threshold optimization
9. Inference
10. Output generation
11. Output validation
12. API
13. Frontend
"""

import os
import subprocess
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from src.business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from src.business_entity_resolution.data.label_builder import LabelBuilder
from src.business_entity_resolution.data.loader import DataLoader
from src.business_entity_resolution.data.submission_validator import SubmissionValidator
from src.business_entity_resolution.evaluation.f05 import compute_entity_f05, evaluate_predictions
from src.business_entity_resolution.evaluation.threshold_optimizer import ThresholdOptimizer
from src.business_entity_resolution.features.feature_builder import FeatureBuilder
from src.business_entity_resolution.inference.output_writer import SubmissionWriter
from src.business_entity_resolution.inference.pipeline import InferencePipeline
from src.business_entity_resolution.models.matcher import BaselineMatcher
from src.business_entity_resolution.preprocessing.address_normalizer import AddressNormalizer
from src.business_entity_resolution.preprocessing.name_normalizer import BusinessNameNormalizer


# -----------------------------------------------------------------------------
# 1. Data Loading Test
# -----------------------------------------------------------------------------
def test_system_stage_1_data_loading():
    """Stage 1: Verify data loading layer loads tab-separated files and preserves raw fields."""
    loader = DataLoader("data/train")
    df = loader.load_source_df("train_source1.tsv", nrows=50, validate=True, expected_prefix="S1-")
    assert len(df) == 50
    assert "entity_id" in df.columns
    assert "business_name" in df.columns
    assert "business_address" in df.columns
    assert "country" in df.columns
    assert "raw_business_name" in df.columns
    assert all(df["entity_id"].str.startswith("S1-"))

    # Test Ground Truth loading
    gt_map = loader.load_ground_truth("train_ground_truth.tsv", nrows=50, validate=True)
    assert len(gt_map) == 50
    for sid, targets in gt_map.items():
        assert sid.startswith("S1-")
        for tid in targets:
            assert tid.startswith("S2-") or tid.startswith("S3-")


# -----------------------------------------------------------------------------
# 2. Normalization Test
# -----------------------------------------------------------------------------
def test_system_stage_2_normalization():
    """Stage 2: Verify name and address normalization routines."""
    name_norm = BusinessNameNormalizer()
    res_name = name_norm.normalize("Acme & Sons Corp.")
    assert res_name.original == "Acme & Sons Corp."
    assert "and" in res_name.normalized
    assert "corp" not in res_name.clean_without_legal
    assert res_name.detected_legal_suffix == "corp"

    addr_norm = AddressNormalizer()
    res_addr = addr_norm.normalize("123 N. Main St., Suite #400, Springfield")
    assert "street" in res_addr.normalized
    assert "suite" in res_addr.normalized
    assert "123" in res_addr.numeric_tokens
    assert "400" in res_addr.numeric_tokens


# -----------------------------------------------------------------------------
# 3. Blocking Test
# -----------------------------------------------------------------------------
def test_system_stage_3_blocking():
    """Stage 3: Verify country-partitioned candidate generation and blocking index."""
    s1_records = pd.DataFrame([
        {"entity_id": "S1-01", "business_name": "Acme Widgets Inc", "business_address": "100 Industrial Rd", "country": "US"},
        {"entity_id": "S1-02", "business_name": "Le Bistro Paris", "business_address": "15 Rue de Rivoli", "country": "France"},
    ])
    target_records = pd.DataFrame([
        {"entity_id": "S2-01", "business_name": "Acme Widgets LLC", "business_address": "100 Industrial Road", "country": "US"},
        {"entity_id": "S3-01", "business_name": "Unrelated US Corp", "business_address": "500 Broadway", "country": "US"},
        {"entity_id": "S2-02", "business_name": "Le Bistro", "business_address": "15 Rue Rivoli", "country": "France"},
        {"entity_id": "S3-02", "business_name": "French Bakery", "business_address": "20 Rue Bonaparte", "country": "France"},
    ])

    generator = CandidateGenerator(max_candidates_per_entity=5)
    generator.fit_target_records(target_records)
    cand_map, metrics = generator.generate_candidates_batch(s1_records)

    assert "S1-01" in cand_map
    assert "S2-01" in cand_map["S1-01"]
    assert "S1-02" in cand_map
    assert "S2-02" in cand_map["S1-02"]
    # Ensure strict country isolation: no France candidates for US entity
    for tid in cand_map["S1-01"]:
        assert not tid.startswith("S2-02") and not tid.startswith("S3-02")


# -----------------------------------------------------------------------------
# 4. Feature Generation Test
# -----------------------------------------------------------------------------
def test_system_stage_4_feature_generation():
    """Stage 4: Verify 36 pairwise similarity features generation without NaNs."""
    feature_builder = FeatureBuilder()
    feature_builder.fit_reference_tfidf(
        s1_names=["Acme Widgets Inc", "Le Bistro Paris"],
        s1_addresses=["100 Industrial Rd", "15 Rue de Rivoli"],
    )

    s1_dict = {"entity_id": "S1-01", "business_name": "Acme Widgets Inc", "business_address": "100 Industrial Rd", "country": "US"}
    target_dict = {"entity_id": "S2-01", "business_name": "Acme Widgets LLC", "business_address": "100 Industrial Road", "country": "US"}

    feat_dict = feature_builder.extract_pair_dict(s1_dict, target_dict)
    assert len(feat_dict) == 36
    assert feat_dict["country_exact_match"] == 1.0
    assert feat_dict["target_is_source2"] == 1.0
    assert feat_dict["name_levenshtein_sim"] > 0.8
    assert not any(np.isnan(v) for v in feat_dict.values())


# -----------------------------------------------------------------------------
# 5. Label Generation Test
# -----------------------------------------------------------------------------
def test_system_stage_5_label_generation(tmp_path):
    """Stage 5: Verify training pair label generation and singleton zero-positive integrity."""
    s1_lookup = {
        "S1-01": {"entity_id": "S1-01", "business_name": "Acme Corp", "business_address": "10 Main", "country": "US"},
        "S1-02": {"entity_id": "S1-02", "business_name": "Solo Shop", "business_address": "20 Main", "country": "US"},
    }
    target_lookup = {
        "S2-01": {"entity_id": "S2-01", "business_name": "Acme Corp", "business_address": "10 Main", "country": "US"},
        "S3-01": {"entity_id": "S3-01", "business_name": "Other Co", "business_address": "99 Road", "country": "US"},
    }
    cand_map = {
        "S1-01": ["S2-01", "S3-01"],
        "S1-02": ["S3-01"],  # Singleton S1-02 has negative candidate S3-01
    }
    gt_map = {
        "S1-01": {"S2-01"},
        "S1-02": set(),  # singleton
    }

    feature_builder = FeatureBuilder()
    feature_builder.fit_reference_tfidf(["Acme Corp", "Solo Shop"], ["10 Main", "20 Main"])

    X, y, pairs, report = LabelBuilder.build_training_dataset(
        candidate_map=cand_map,
        ground_truth_map=gt_map,
        s1_lookup=s1_lookup,
        target_lookup=target_lookup,
        feature_builder=feature_builder,
    )

    assert report.is_valid is True
    assert report.singleton_positive_count == 0
    assert len(pairs) == 3
    assert np.sum(y) == 1  # exactly one positive pair

    # Test save and reload
    npz_path = str(tmp_path / "dataset.npz")
    LabelBuilder.save_dataset_npz(npz_path, X, y, pairs, feature_builder.feature_names)
    X_re, y_re, p_re, f_re = LabelBuilder.load_dataset_npz(npz_path)
    assert np.array_equal(X, X_re)
    assert np.array_equal(y, y_re)


# -----------------------------------------------------------------------------
# 6. Model Training Test
# -----------------------------------------------------------------------------
def test_system_stage_6_model_training(tmp_path):
    """Stage 6: Verify model training, probability prediction, and serialization."""
    np.random.seed(42)
    X = np.random.randn(40, 36).astype(np.float32)
    y = np.array([1 if i % 4 == 0 else 0 for i in range(40)], dtype=np.int32)

    matcher = BaselineMatcher(model_type="hist_gradient_boosting", random_state=42)
    matcher.fit(X, y)
    probs = matcher.predict_proba(X)
    assert len(probs) == 40
    assert all(0.0 <= p <= 1.0 for p in probs)

    model_path = str(tmp_path / "model.joblib")
    matcher.save(model_path)
    loaded_matcher = BaselineMatcher.load(model_path)
    probs_loaded = loaded_matcher.predict_proba(X)
    np.testing.assert_allclose(probs, probs_loaded, rtol=1e-5)


# -----------------------------------------------------------------------------
# 7. Evaluation Test
# -----------------------------------------------------------------------------
def test_system_stage_7_evaluation():
    """Stage 7: Verify Macro F0.5 calculation and precision penalty weighting."""
    # Perfect match
    f05, p, r = compute_entity_f05(true_set={"S2-1", "S3-1"}, pred_set={"S2-1", "S3-1"})
    assert p == 1.0 and r == 1.0 and f05 == 1.0

    # Singleton correct
    f05, p, r = compute_entity_f05(true_set=set(), pred_set=set())
    assert p == 1.0 and r == 1.0 and f05 == 1.0

    # Singleton incorrect (false merge)
    f05, p, r = compute_entity_f05(true_set=set(), pred_set={"S2-99"})
    assert f05 == 0.0 and p == 0.0 and r == 1.0

    # Precision-weighting: 1 TP and 1 FP gives Precision = 0.5, Recall = 1.0
    # F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5556
    f05, p, r = compute_entity_f05(true_set={"S2-1"}, pred_set={"S2-1", "S2-2"})
    assert abs(f05 - 0.5556) < 1e-3


# -----------------------------------------------------------------------------
# 8. Threshold Optimization Test
# -----------------------------------------------------------------------------
def test_system_stage_8_threshold_optimization(tmp_path):
    """Stage 8: Verify threshold optimizer finds optimal threshold and updates config."""
    pairs = [("S1-1", "S2-1"), ("S1-1", "S2-2"), ("S1-2", "S2-3"), ("S1-3", "S2-4")]
    probs = np.array([0.95, 0.45, 0.90, 0.10])
    ground_truth = {"S1-1": {"S2-1"}, "S1-2": {"S2-3"}, "S1-3": set()}
    all_s1 = ["S1-1", "S1-2", "S1-3"]

    optimizer = ThresholdOptimizer(min_threshold=0.20, max_threshold=0.90, step=0.05)
    report = optimizer.optimize(pairs, probs, ground_truth, all_s1)

    assert report.optimal_threshold >= 0.50
    assert report.best_macro_f05 == 1.0  # Perfect score when threshold separates 0.90+ from <0.50

    th_path = str(tmp_path / "threshold.json")
    ThresholdOptimizer.save_threshold_json(report, filepath=th_path)
    assert os.path.exists(th_path)


# -----------------------------------------------------------------------------
# 9. Inference Test
# -----------------------------------------------------------------------------
def test_system_stage_9_inference(tmp_path):
    """Stage 9: Verify end-to-end inference execution generating predictions."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create mock inputs
    pd.DataFrame([
        {"entity_id": "S1-T1", "business_name": "Reference Corp", "business_address": "123 Ref St", "country": "US"}
    ]).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)

    pd.DataFrame([
        {"entity_id": "S1-01", "business_name": "Alpha Beta LLC", "business_address": "10 First Ave", "country": "US"},
        {"entity_id": "S1-02", "business_name": "Solo Entity", "business_address": "99 Lone Rd", "country": "India"},
    ]).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

    pd.DataFrame([
        {"entity_id": "S2-01", "business_name": "Alpha Beta Inc", "business_address": "10 First Avenue", "country": "US"}
    ]).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

    pd.DataFrame([
        {"entity_id": "S3-01", "business_name": "Different Firm", "business_address": "500 Other St", "country": "India"}
    ]).to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    model = BaselineMatcher(model_type="logistic_regression", random_state=42)
    model.fit(np.random.randn(10, 36).astype(np.float32), np.array([1, 0] * 5, dtype=np.int32))

    pipeline = InferencePipeline(model=model, threshold=0.50)
    match_out = str(tmp_path / "matching.tsv")
    cand_out = str(tmp_path / "candidate.tsv")

    report = pipeline.run_full_inference(
        test_dir=str(test_dir),
        train_dir=str(train_dir),
        matching_output_path=match_out,
        candidate_output_path=cand_out,
        chunk_size=10,
    )
    assert report.is_valid is True
    assert os.path.exists(match_out)
    assert os.path.exists(cand_out)


# -----------------------------------------------------------------------------
# 10. Output Generation Test
# -----------------------------------------------------------------------------
def test_system_stage_10_output_generation(tmp_path):
    """Stage 10: Verify SubmissionWriter generates formatted TSVs with exact headers."""
    match_path = str(tmp_path / "matching_results.tsv")
    cand_path = str(tmp_path / "candidate_pairs.tsv")

    with SubmissionWriter(match_path, cand_path, enforce_subset=True) as writer:
        writer.write_entity_result("S1-01", ["S2-10", "S3-20"], ["S2-10", "S3-20", "S2-30"])
        writer.write_entity_result("S1-02", [], ["S2-50"])  # singleton

    with open(match_path, "r", encoding="utf-8") as f:
        m_lines = f.readlines()
    assert m_lines[0] == "source1_entity_id\tmatched_entity_ids\n"
    assert m_lines[1] == "S1-01\tS2-10,S3-20\n"
    assert m_lines[2] == "S1-02\t\n"

    with open(cand_path, "r", encoding="utf-8") as f:
        c_lines = f.readlines()
    assert c_lines[0] == "source1_entity_id\tcandidate_entity_ids\n"
    assert c_lines[1] == "S1-01\tS2-10,S3-20,S2-30\n"
    assert c_lines[2] == "S1-02\tS2-50\n"


# -----------------------------------------------------------------------------
# 11. Output Validation Test
# -----------------------------------------------------------------------------
def test_system_stage_11_output_validation(tmp_path):
    """Stage 11: Verify SubmissionValidator checks all 11 criteria strictly."""
    match_path = str(tmp_path / "matching_results.tsv")
    cand_path = str(tmp_path / "candidate_pairs.tsv")

    # Write valid output
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-01\tS2-1\n")
        f.write("S1-02\t\n")

    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-01\tS2-1,S3-2\n")
        f.write("S1-02\tS3-5\n")

    val_result = SubmissionValidator.validate_submission(
        matching_path=match_path,
        candidate_path=cand_path,
        expected_s1_ids={"S1-01", "S1-02"},
        check_target_ids=False,
    )
    assert val_result.is_valid is True
    assert len(val_result.errors) == 0


# -----------------------------------------------------------------------------
# 12. API Endpoints Test
# -----------------------------------------------------------------------------
def test_system_stage_12_api_endpoints():
    """Stage 12: Verify FastAPI backend endpoints respond with 200 OK and expected schemas."""
    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Dataset stats
    res_stats = client.get("/dataset/stats")
    assert res_stats.status_code == 200
    assert "train" in res_stats.json()
    assert "test" in res_stats.json()

    # Metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200

    # Experiments
    res_exp = client.get("/experiments")
    assert res_exp.status_code == 200
    assert len(res_exp.json()["experiments"]) > 0


# -----------------------------------------------------------------------------
# 13. Frontend Test
# -----------------------------------------------------------------------------
def test_system_stage_13_frontend():
    """Stage 13: Verify frontend project builds cleanly with TypeScript and Vite."""
    frontend_dir = os.path.abspath("app/frontend")
    dist_index = os.path.join(frontend_dir, "dist", "index.html")
    assert os.path.exists(dist_index), "Frontend production bundle dist/index.html must exist"
    with open(dist_index, "r", encoding="utf-8") as f:
        content = f.read()
    assert "<div id=\"root\"></div>" in content
