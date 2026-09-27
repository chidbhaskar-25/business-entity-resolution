"""Unit tests for F0.5 metric implementation against manually calculated examples."""

import pytest
from src.business_entity_resolution.evaluation.f05 import (
    compute_entity_f05,
    compute_macro_f05,
    evaluate_entity_records,
)


def test_f05_perfect_match():
    """T = {S2-1, S3-1}, P = {S2-1, S3-1} -> P=1.0, R=1.0, F0.5=1.0"""
    true_set = {"S2-1", "S3-1"}
    pred_set = {"S2-1", "S3-1"}
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert p == 1.0
    assert r == 1.0
    assert f05 == 1.0


def test_f05_false_negative_penalty():
    """T = {S2-1, S3-1}, P = {S2-1} -> P=1.0, R=0.5
    F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.8333333333333334
    """
    true_set = {"S2-1", "S3-1"}
    pred_set = {"S2-1"}
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert p == 1.0
    assert r == 0.5
    assert pytest.approx(f05, rel=1e-5) == 5.0 / 6.0


def test_f05_false_positive_heavy_penalty():
    """T = {S2-1}, P = {S2-1, S3-2} -> P=0.5, R=1.0
    F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.5555555555555556
    Note that 5/9 (0.5556) is significantly lower than 5/6 (0.8333),
    confirming F0.5 heavily penalizes false positives over false negatives.
    """
    true_set = {"S2-1"}
    pred_set = {"S2-1", "S3-2"}
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert p == 0.5
    assert r == 1.0
    assert pytest.approx(f05, rel=1e-5) == 5.0 / 9.0


def test_f05_balanced_tp_fp_fn():
    """T = {A, B, C}, P = {A, B, D} -> TP=2, P=2/3, R=2/3
    F0.5 = (1.25 * 2/3 * 2/3) / (0.25 * 2/3 + 2/3) = (1.25 * 4/9) / (1.25 * 2/3) = 2/3 = 0.6666666666666666
    """
    true_set = {"A", "B", "C"}
    pred_set = {"A", "B", "D"}
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert pytest.approx(p, rel=1e-5) == 2.0 / 3.0
    assert pytest.approx(r, rel=1e-5) == 2.0 / 3.0
    assert pytest.approx(f05, rel=1e-5) == 2.0 / 3.0


def test_f05_zero_tp_completely_wrong():
    """T = {S2-1}, P = {S2-99} -> TP=0, P=0.0, R=0.0, F0.5=0.0"""
    true_set = {"S2-1"}
    pred_set = {"S2-99"}
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert p == 0.0
    assert r == 0.0
    assert f05 == 0.0


def test_f05_correct_singleton():
    """T = {}, P = {} -> Real-world singleton correctly predicted with 0 links.
    Competition rule: P=1.0, R=1.0, F0.5=1.0
    """
    true_set = set()
    pred_set = set()
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert p == 1.0
    assert r == 1.0
    assert f05 == 1.0


def test_f05_incorrect_singleton():
    """T = {}, P = {S2-500} -> Singleton incorrectly assigned false match.
    Competition rule: P=0.0, R=1.0, F0.5=0.0
    """
    true_set = set()
    pred_set = {"S2-500"}
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert p == 0.0
    assert r == 1.0
    assert f05 == 0.0


def test_f05_missed_matches_empty_prediction():
    """T = {S2-1}, P = {} -> Missed match completely.
    Competition rule: P=1.0, R=0.0, F0.5=0.0
    """
    true_set = {"S2-1"}
    pred_set = set()
    f05, p, r = compute_entity_f05(true_set, pred_set)
    assert p == 1.0
    assert r == 0.0
    assert f05 == 0.0


def test_macro_f05_four_entities_manual_calculation():
    """Macro-average across 4 distinct entity types:
    Entity 1: T={A, B}, P={A} -> F0.5 = 5/6
    Entity 2: T={A}, P={A, C} -> F0.5 = 5/9
    Entity 3: T={}, P={} -> Singleton correct -> F0.5 = 1.0
    Entity 4: T={}, P={D} -> Singleton corrupted -> F0.5 = 0.0

    Expected Macro F0.5 = (5/6 + 5/9 + 1.0 + 0.0) / 4
                        = (15/18 + 10/18 + 18/18 + 0) / 4
                        = (43/18) / 4 = 43/72 = 0.5972222222222222
    """
    gt = {
        "S1-1": {"A", "B"},
        "S1-2": {"A"},
        "S1-3": set(),
        "S1-4": set(),
    }
    preds = {
        "S1-1": {"A"},
        "S1-2": {"A", "C"},
        "S1-3": set(),
        "S1-4": {"D"},
    }
    all_s1 = ["S1-1", "S1-2", "S1-3", "S1-4"]

    summary, records = evaluate_entity_records(gt, preds, all_s1)

    expected_macro = 43.0 / 72.0
    assert pytest.approx(summary.macro_f05, rel=1e-4) == expected_macro
    assert summary.total_entities == 4
    assert summary.singleton_entities == 2
    assert summary.singleton_accuracy == 0.5  # 1 out of 2 correct
    assert summary.non_singleton_entities == 2
    assert len(records) == 4


def test_competition_evaluator_mock_model(tmp_path):
    import numpy as np
    from src.business_entity_resolution.evaluation.metrics import CompetitionEvaluator
    from src.business_entity_resolution.models.matcher import BaselineMatcher

    # Train a simple mock matcher
    model_path = str(tmp_path / "mock_matcher.joblib")
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    y = np.array([1, 0, 1], dtype=np.int32)
    matcher = BaselineMatcher(model_type="logistic_regression", random_state=42)
    matcher.fit(X, y)
    matcher.save(model_path)

    # Evaluate candidates
    evaluator = CompetitionEvaluator(model_path=model_path, threshold=0.50)
    pairs = [("S1-01", "S2-101"), ("S1-01", "S3-102"), ("S1-02", "S2-201")]
    gt = {"S1-01": {"S2-101"}, "S1-02": {"S2-201"}}
    summary, records = evaluator.evaluate_candidates(pairs, X, gt, all_s1_ids=["S1-01", "S1-02", "S1-03"])

    assert summary.total_entities == 3
    assert summary.singleton_entities == 1  # S1-03 is singleton
    assert summary.singleton_accuracy == 1.0  # S1-03 had 0 predictions
    assert summary.macro_f05 > 0.0

