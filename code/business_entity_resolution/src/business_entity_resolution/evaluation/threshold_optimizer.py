"""Threshold optimization for Entity Resolution maximizing Macro F0.5."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import yaml
from pydantic import BaseModel
from sklearn.model_selection import GroupShuffleSplit

from ..data.label_builder import LabelBuilder
from ..models.matcher import BaselineMatcher
from ..utils.logger import setup_logger
from .f05 import compute_entity_f05

logger = setup_logger("evaluation.threshold_optimizer")


class ThresholdStepMetrics(BaseModel):
    """Evaluation metrics for a specific decision threshold value."""
    threshold: float
    macro_f05: float
    macro_precision: float
    macro_recall: float
    total_predicted_matches: int
    singleton_correct_count: int
    singleton_accuracy: float
    non_singleton_macro_f05: float


class ThresholdOptimizationReport(BaseModel):
    """Comprehensive threshold search experiment report."""
    optimal_threshold: float
    best_macro_f05: float
    best_macro_precision: float
    best_macro_recall: float
    default_threshold: float
    default_macro_f05: float
    f05_improvement: float
    total_entities_evaluated: int
    singleton_entities: int
    step_curve: List[ThresholdStepMetrics]


class ThresholdOptimizer:
    """Explores decision thresholds and selects optimal threshold maximizing Macro F0.5."""

    def __init__(
        self,
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        step: float = 0.02,
        default_threshold: float = 0.50,
    ):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.step = step
        self.default_threshold = default_threshold

    def optimize(
        self,
        pairs: List[Tuple[str, str]],
        probabilities: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: List[str],
    ) -> ThresholdOptimizationReport:
        """Evaluates thresholds across range and identifies optimal threshold.

        For each threshold:
        - Calculates Macro Precision, Recall, and F0.5 across all S1 entities
        - Calculates total predicted matches
        - Calculates singleton predictions & accuracy
        - Records metrics in search curve

        Args:
            pairs: List of (s1_id, target_id) candidate tuples.
            probabilities: Model predicted probabilities for pairs.
            ground_truth: Mapping s1_id -> Set[target_id].
            all_s1_ids: All S1 entities in evaluation split (including singletons).

        Returns:
            ThresholdOptimizationReport with optimal threshold and search curve.
        """
        logger.info(f"Grouping candidate predictions for {len(all_s1_ids):,} Source 1 entities...")
        cands_by_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (s1_id, target_id), prob in zip(pairs, probabilities):
            cands_by_s1[s1_id].append((target_id, float(prob)))

        # Pre-count singletons
        singleton_s1_ids = {s1 for s1 in all_s1_ids if not ground_truth.get(s1)}
        total_singletons = len(singleton_s1_ids)

        thresholds = np.arange(self.min_threshold, self.max_threshold + 1e-5, self.step)
        curve: List[ThresholdStepMetrics] = []

        best_th = self.default_threshold
        best_f05 = -1.0
        best_p = 0.0
        best_r = 0.0
        default_f05 = 0.0

        logger.info(
            f"Scanning {len(thresholds)} thresholds in range [{self.min_threshold:.2f}, {self.max_threshold:.2f}] "
            f"with step={self.step:.2f}..."
        )

        for th in thresholds:
            th_val = round(float(th), 4)

            f05_scores: List[float] = []
            p_scores: List[float] = []
            r_scores: List[float] = []
            non_sing_f05: List[float] = []

            total_pred_matches = 0
            correct_singletons = 0

            for s1_id in all_s1_ids:
                t_set = ground_truth.get(s1_id, set())
                cands = cands_by_s1.get(s1_id, [])

                # Filter predicted matches where probability >= threshold
                p_set = {tid for tid, p in cands if p >= th_val}
                total_pred_matches += len(p_set)

                f05, p, r = compute_entity_f05(t_set, p_set)
                f05_scores.append(f05)
                p_scores.append(p)
                r_scores.append(r)

                if not t_set:
                    if not p_set:
                        correct_singletons += 1
                else:
                    non_sing_f05.append(f05)

            macro_f05 = float(np.mean(f05_scores))
            macro_p = float(np.mean(p_scores))
            macro_r = float(np.mean(r_scores))
            sing_acc = (correct_singletons / total_singletons) if total_singletons > 0 else 1.0
            non_sing_macro = float(np.mean(non_sing_f05)) if non_sing_f05 else 0.0

            step_metric = ThresholdStepMetrics(
                threshold=th_val,
                macro_f05=round(macro_f05, 5),
                macro_precision=round(macro_p, 5),
                macro_recall=round(macro_r, 5),
                total_predicted_matches=total_pred_matches,
                singleton_correct_count=correct_singletons,
                singleton_accuracy=round(sing_acc, 5),
                non_singleton_macro_f05=round(non_sing_macro, 5),
            )
            curve.append(step_metric)

            if abs(th_val - self.default_threshold) < (self.step / 2.0 + 1e-5):
                default_f05 = macro_f05

            # Select higher F0.5; if tied within 1e-5, break ties by precision then proximity to 0.50
            is_better = False
            if macro_f05 > best_f05 + 1e-5:
                is_better = True
            elif abs(macro_f05 - best_f05) <= 1e-5:
                if macro_p > best_p + 1e-5:
                    is_better = True
                elif abs(macro_p - best_p) <= 1e-5:
                    if abs(th_val - self.default_threshold) < abs(best_th - self.default_threshold):
                        is_better = True

            if is_better:
                best_f05 = macro_f05
                best_th = th_val
                best_p = macro_p
                best_r = macro_r

        improvement = best_f05 - default_f05

        logger.info(
            f"Optimization finished! Optimal threshold: {best_th:.2f} "
            f"(Macro F0.5: {best_f05:.4f} vs default 0.50: {default_f05:.4f}, diff={improvement:+.4f})"
        )

        return ThresholdOptimizationReport(
            optimal_threshold=best_th,
            best_macro_f05=round(best_f05, 5),
            best_macro_precision=round(best_p, 5),
            best_macro_recall=round(best_r, 5),
            default_threshold=self.default_threshold,
            default_macro_f05=round(default_f05, 5),
            f05_improvement=round(improvement, 5),
            total_entities_evaluated=len(all_s1_ids),
            singleton_entities=total_singletons,
            step_curve=curve,
        )

    @staticmethod
    def save_threshold_json(
        report: ThresholdOptimizationReport,
        filepath: str = "models/threshold.json",
    ) -> None:
        """Saves optimal threshold and key metrics to models/threshold.json."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "optimal_threshold": report.optimal_threshold,
            "best_macro_f05": report.best_macro_f05,
            "best_macro_precision": report.best_macro_precision,
            "best_macro_recall": report.best_macro_recall,
            "default_macro_f05": report.default_macro_f05,
            "f05_improvement": report.f05_improvement,
            "total_entities_evaluated": report.total_entities_evaluated,
            "singleton_entities": report.singleton_entities,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved optimal threshold configuration to: {filepath}")

    @staticmethod
    def update_config_yaml(
        optimal_threshold: float,
        config_path: str = "config.yaml",
    ) -> None:
        """Updates or appends selected optimal threshold in config.yaml."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, skipping update.")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "threshold_optimization" not in cfg:
            cfg["threshold_optimization"] = {}

        cfg["threshold_optimization"]["selected_optimal_threshold"] = optimal_threshold

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated config.yaml with selected_optimal_threshold={optimal_threshold}")

    @staticmethod
    def save_experiment_report(
        report: ThresholdOptimizationReport,
        filepath: str = "outputs/threshold_experiment.json",
    ) -> None:
        """Saves full step-by-step threshold sweep curve to outputs/threshold_experiment.json."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        logger.info(f"Saved threshold experiment curve to: {filepath}")


def run_threshold_optimization_pipeline(
    dataset_path: str = "outputs/train_dataset.npz",
    model_path: str = "models/baseline_model.joblib",
    config_path: str = "config.yaml",
    threshold_json_path: str = "models/threshold.json",
    experiment_report_path: str = "outputs/threshold_experiment.json",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.02,
    use_validation_split: bool = True,
    val_size: float = 0.20,
    random_seed: int = 42,
) -> ThresholdOptimizationReport:
    """Executes threshold optimization search and persists results."""
    logger.info("=" * 75)
    logger.info("STARTING THRESHOLD OPTIMIZATION SEARCH (MAXIMIZING MACRO F0.5)")
    logger.info("=" * 75)

    # 1. Load trained model
    matcher = BaselineMatcher.load(model_path)

    # 2. Load dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    X, y, pairs, feature_names = LabelBuilder.load_dataset_npz(dataset_path)

    # 3. Partition into validation split if requested (prevents threshold overfitting)
    s1_ids = np.array([p[0] for p in pairs])
    if use_validation_split:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_seed)
        _, eval_idx = next(gss.split(X, y, groups=s1_ids))
        logger.info(f"Using holdout validation split: {len(eval_idx):,} pairs ({len(np.unique(s1_ids[eval_idx])):,} S1 entities)")
    else:
        eval_idx = np.arange(len(pairs))
        logger.info(f"Using full dataset: {len(eval_idx):,} pairs ({len(np.unique(s1_ids)):,} S1 entities)")

    X_eval = X[eval_idx]
    pairs_eval = [pairs[i] for i in eval_idx]
    y_eval = y[eval_idx]

    # Reconstruct ground truth mapping for evaluation entities
    ground_truth: Dict[str, Set[str]] = defaultdict(set)
    eval_s1_set = set()
    for (s1_id, target_id), label in zip(pairs_eval, y_eval):
        eval_s1_set.add(s1_id)
        if label == 1:
            ground_truth[s1_id].add(target_id)

    for s1_id in eval_s1_set:
        if s1_id not in ground_truth:
            ground_truth[s1_id] = set()

    all_s1_ids = sorted(list(eval_s1_set))

    # 4. Predict probabilities
    logger.info(f"Computing candidate probabilities for {len(pairs_eval):,} pairs...")
    probs = matcher.predict_proba(X_eval)

    # 5. Run threshold optimization search
    optimizer = ThresholdOptimizer(
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
        default_threshold=0.50,
    )
    report = optimizer.optimize(
        pairs=pairs_eval,
        probabilities=probs,
        ground_truth=ground_truth,
        all_s1_ids=all_s1_ids,
    )

    # 6. Save results
    optimizer.save_threshold_json(report, threshold_json_path)
    optimizer.update_config_yaml(report.optimal_threshold, config_path)
    optimizer.save_experiment_report(report, experiment_report_path)

    # 7. Print formatted experiment table
    print("\n" + "=" * 90)
    print("THRESHOLD OPTIMIZATION EXPERIMENT: MACRO F0.5 SWEEP REPORT")
    print("=" * 90)
    print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>9} | {'Pred Matches':>13} | {'Singleton Acc':>14}")
    print("-" * 90)

    min_def_diff = min(abs(r.threshold - 0.50) for r in report.step_curve)

    # Sample representative rows across curve (including default 0.50 and optimal)
    for row in report.step_curve:
        is_opt = abs(row.threshold - report.optimal_threshold) < 1e-4
        is_def = abs(abs(row.threshold - 0.50) - min_def_diff) < 1e-5
        flag = "  <-- OPTIMAL" if is_opt else ("  <-- DEFAULT (0.50)" if is_def else "")
        if is_opt or is_def or (round(row.threshold * 100) % 10 == 0):
            print(
                f"{row.threshold:10.2f} | {row.macro_f05:11.4f} | {row.macro_precision:10.4f} | "
                f"{row.macro_recall:9.4f} | {row.total_predicted_matches:13,d} | {row.singleton_accuracy:13.2%}{flag}"
            )

    print("-" * 90)
    print(f"Optimal Selected Threshold:   {report.optimal_threshold:.2f}")
    print(f"Validation Macro F0.5 @ opt:  {report.best_macro_f05:.4f} (vs @ 0.50: {report.default_macro_f05:.4f}, diff: {report.f05_improvement:+.4f})")
    print(f"Validation Precision @ opt:   {report.best_macro_precision:.4f}")
    print(f"Validation Recall @ opt:      {report.best_macro_recall:.4f}")
    print(f"Threshold Saved To:           {threshold_json_path}")
    print(f"Config Updated:               {config_path}")
    print(f"Experiment Curve Saved To:    {experiment_report_path}")
    print("=" * 90 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize probability decision threshold for Macro F0.5")
    parser.add_argument("--dataset-path", type=str, default="outputs/train_dataset.npz")
    parser.add_argument("--model-path", type=str, default="models/baseline_model.joblib")
    parser.add_argument("--config-path", type=str, default="config.yaml")
    parser.add_argument("--threshold-json", type=str, default="models/threshold.json")
    parser.add_argument("--experiment-output", type=str, default="outputs/threshold_experiment.json")
    parser.add_argument("--min-th", type=float, default=0.05)
    parser.add_argument("--max-th", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--full-dataset", action="store_true", help="Optimize on full dataset instead of validation split")
    args = parser.parse_args()

    run_threshold_optimization_pipeline(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        config_path=args.config_path,
        threshold_json_path=args.threshold_json,
        experiment_report_path=args.experiment_output,
        min_threshold=args.min_th,
        max_threshold=args.max_th,
        step=args.step,
        use_validation_split=not args.full_dataset,
    )
