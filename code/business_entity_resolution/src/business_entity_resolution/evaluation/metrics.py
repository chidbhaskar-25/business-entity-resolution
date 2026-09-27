"""Competition-style entity-level evaluation pipeline for Business Entity Resolution."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..data.label_builder import LabelBuilder
from ..models.matcher import BaselineMatcher
from ..utils.logger import setup_logger
from .f05 import (
    EntityMetricRecord,
    EntityResolutionMetrics,
    MacroEvaluationSummary,
    compute_entity_f05,
    compute_macro_f05,
    evaluate_entity_records,
    evaluate_predictions,
)

logger = setup_logger("evaluation.metrics")


class CompetitionEvaluator:
    """Evaluates entity resolution predictions strictly at the Source 1 entity level."""

    def __init__(
        self,
        model_path: str = "models/baseline_model.joblib",
        threshold: float = 0.50,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.matcher = BaselineMatcher.load(model_path)

    def evaluate_candidates(
        self,
        pairs: List[Tuple[str, str]],
        X: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[MacroEvaluationSummary, List[EntityMetricRecord]]:
        """Executes full Source 1 entity-level evaluation.

        For each Source 1 entity:
        1. Identifies candidate pairs
        2. Scores candidate probabilities
        3. Applies probability threshold
        4. Creates predicted match list
        5. Compares with ground truth
        6. Calculates precision
        7. Calculates recall
        8. Calculates F0.5
        Then macro-averages across all Source 1 entities (including singletons).

        Args:
            pairs: List of (s1_id, target_id) candidate pairs.
            X: Corresponding pairwise feature matrix.
            ground_truth: Ground truth mapping s1_id -> Set[target_id].
            all_s1_ids: Comprehensive list of Source 1 entities (singletons included).
            threshold: Probability decision threshold (defaults to self.threshold).

        Returns:
            Tuple of (MacroEvaluationSummary, List[EntityMetricRecord]).
        """
        th = threshold if threshold is not None else self.threshold
        logger.info(f"Scoring {len(pairs):,} candidate pairs using {self.matcher.model_type} (th={th:.3f})...")

        # 1 & 2. Score candidates
        probs = self.matcher.predict_proba(X)

        # 3 & 4. Group candidates by S1 entity and apply threshold to create predicted match list
        predictions: Dict[str, Set[str]] = defaultdict(set)

        # Identify all S1 IDs present in pairs or ground truth
        s1_universe = set(ground_truth.keys())
        for s1_id, _ in pairs:
            s1_universe.add(s1_id)
        if all_s1_ids:
            s1_universe.update(all_s1_ids)

        # Initialize all S1 entities with empty predictions
        for s1_id in s1_universe:
            predictions[s1_id] = set()

        for (s1_id, target_id), prob in zip(pairs, probs):
            if prob >= th:
                predictions[s1_id].add(target_id)

        eval_ids = sorted(list(s1_universe))
        logger.info(f"Evaluating {len(eval_ids):,} Source 1 entities at decision threshold={th:.3f}...")

        # 5, 6, 7, 8 & Macro-average
        summary, entity_records = evaluate_entity_records(
            ground_truth=ground_truth,
            predictions=predictions,
            all_s1_ids=eval_ids,
        )

        return summary, entity_records


def run_evaluation(
    dataset_path: str = "outputs/train_dataset.npz",
    model_path: str = "models/baseline_model.joblib",
    threshold: float = 0.50,
    output_path: str = "outputs/metrics.json",
) -> Dict[str, Any]:
    """Runs Source 1 entity-level evaluation and saves outputs/metrics.json."""
    logger.info("=" * 75)
    logger.info("RUNNING COMPETITION-STYLE ENTITY-LEVEL EVALUATION")
    logger.info("=" * 75)

    evaluator = CompetitionEvaluator(model_path=model_path, threshold=threshold)

    logger.info(f"Loading dataset from: {dataset_path}")
    X, y, pairs, feature_names = LabelBuilder.load_dataset_npz(dataset_path)

    # Reconstruct ground truth map from dataset labels
    ground_truth: Dict[str, Set[str]] = defaultdict(set)
    all_s1_set = set()
    for (s1_id, target_id), label in zip(pairs, y):
        all_s1_set.add(s1_id)
        if label == 1:
            ground_truth[s1_id].add(target_id)

    # Also make sure singletons in all_s1_set have empty sets in ground_truth
    for s1_id in all_s1_set:
        if s1_id not in ground_truth:
            ground_truth[s1_id] = set()

    all_s1_ids = sorted(list(all_s1_set))

    summary, entity_records = evaluator.evaluate_candidates(
        pairs=pairs,
        X=X,
        ground_truth=ground_truth,
        all_s1_ids=all_s1_ids,
        threshold=threshold,
    )

    # Package output payload
    output_dict = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "decision_threshold": threshold,
        "macro_metrics": summary.model_dump(),
        "sample_entity_evaluations": [r.model_dump() for r in entity_records[:20]],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    logger.info(f"Saved evaluation metrics to: {output_path}")

    # Display clear formatted report
    print("\n" + "=" * 75)
    print("AMAZON ML CHALLENGE 2026: ENTITY RESOLUTION EVALUATION")
    print("=" * 75)
    print(f"Total Source 1 Entities:     {summary.total_entities:,}")
    print(f"Singleton Entities (No Match):{summary.singleton_entities:,} ({summary.singleton_entities / summary.total_entities:.2%})")
    print(f"Singleton Accuracy:          {summary.singleton_accuracy:.2%}")
    print(f"Non-Singleton Entities:      {summary.non_singleton_entities:,}")
    print(f"Non-Singleton Macro F0.5:    {summary.non_singleton_macro_f05:.4f}")
    print("-" * 75)
    print(f"OVERALL MACRO F0.5 SCORE:    {summary.macro_f05:.4f}")
    print(f"Overall Macro Precision:     {summary.macro_precision:.4f}")
    print(f"Overall Macro Recall:        {summary.macro_recall:.4f}")
    print("-" * 75)
    print(f"Total True Target Matches:   {summary.total_true_matches:,}")
    print(f"Total Predicted Matches:     {summary.total_predicted_matches:,}")
    print(f"Total True Positives:        {summary.total_true_positives:,}")
    print(f"Micro-Averaged Precision:    {summary.micro_precision:.4f}")
    print(f"Micro-Averaged Recall:       {summary.micro_recall:.4f}")
    print("-" * 75)
    print(f"Report Output Saved To:      {output_path}")
    print("=" * 75 + "\n")

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate entity resolution model at S1 entity level")
    parser.add_argument("--dataset-path", type=str, default="outputs/train_dataset.npz")
    parser.add_argument("--model-path", type=str, default="models/baseline_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output", type=str, default="outputs/metrics.json")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        threshold=args.threshold,
        output_path=args.output,
    )
