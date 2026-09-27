"""Unit tests for threshold optimization pipeline and persistence."""

import os
import tempfile
import numpy as np
import pytest
import yaml

from src.business_entity_resolution.evaluation.threshold_optimizer import (
    ThresholdOptimizationReport,
    ThresholdOptimizer,
    ThresholdStepMetrics,
)


def test_threshold_optimizer_synthetic_sweep():
    # 3 entities:
    # S1-1 has high prob candidate (0.90) and low prob candidate (0.30), true match is S2-A
    # S1-2 has true match S2-B with prob 0.70
    # S1-3 is singleton with a false candidate at prob 0.20
    pairs = [
        ("S1-1", "S2-A"),  # True match
        ("S1-1", "S2-X"),  # Non-match
        ("S1-2", "S2-B"),  # True match
        ("S1-3", "S2-Y"),  # Non-match on singleton
    ]
    probs = np.array([0.90, 0.30, 0.70, 0.20], dtype=np.float32)
    gt = {
        "S1-1": {"S2-A"},
        "S1-2": {"S2-B"},
        "S1-3": set(),
    }
    all_s1 = ["S1-1", "S1-2", "S1-3"]

    optimizer = ThresholdOptimizer(min_threshold=0.10, max_threshold=0.95, step=0.05)
    report = optimizer.optimize(pairs, probs, gt, all_s1)

    assert isinstance(report, ThresholdOptimizationReport)
    assert len(report.step_curve) > 5
    assert report.total_entities_evaluated == 3
    assert report.singleton_entities == 1

    # Verify that every step contains all required metrics
    for step in report.step_curve:
        assert 0.0 <= step.macro_f05 <= 1.0
        assert 0.0 <= step.macro_precision <= 1.0
        assert 0.0 <= step.macro_recall <= 1.0
        assert step.total_predicted_matches >= 0
        assert 0 <= step.singleton_correct_count <= 1
        assert 0.0 <= step.singleton_accuracy <= 1.0

    # At threshold 0.50:
    # S1-1 predicts S2-A (true match) -> P=1, R=1, F0.5=1.0
    # S1-2 predicts S2-B (true match) -> P=1, R=1, F0.5=1.0
    # S1-3 predicts nothing (correct singleton) -> F0.5=1.0
    # Macro F0.5 at 0.50 should be 1.0!
    # And at threshold 0.10: S1-3 predicts S2-Y (singleton error) -> Macro F0.5 drops!
    assert report.best_macro_f05 == 1.0
    assert 0.40 <= report.optimal_threshold <= 0.70


def test_threshold_optimizer_persistence_and_config_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "threshold.json")
        exp_path = os.path.join(tmpdir, "experiment.json")
        config_path = os.path.join(tmpdir, "config.yaml")

        # Create dummy config.yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"threshold_optimization": {"default_threshold": 0.5}}, f)

        # Mock report
        step = ThresholdStepMetrics(
            threshold=0.62,
            macro_f05=0.965,
            macro_precision=0.960,
            macro_recall=0.970,
            total_predicted_matches=100,
            singleton_correct_count=10,
            singleton_accuracy=0.90,
            non_singleton_macro_f05=0.97,
        )
        report = ThresholdOptimizationReport(
            optimal_threshold=0.62,
            best_macro_f05=0.965,
            best_macro_precision=0.960,
            best_macro_recall=0.970,
            default_threshold=0.50,
            default_macro_f05=0.940,
            f05_improvement=0.025,
            total_entities_evaluated=50,
            singleton_entities=10,
            step_curve=[step],
        )

        ThresholdOptimizer.save_threshold_json(report, json_path)
        assert os.path.exists(json_path)

        ThresholdOptimizer.save_experiment_report(report, exp_path)
        assert os.path.exists(exp_path)

        ThresholdOptimizer.update_config_yaml(report.optimal_threshold, config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            updated_cfg = yaml.safe_load(f)
        assert updated_cfg["threshold_optimization"]["selected_optimal_threshold"] == 0.62
