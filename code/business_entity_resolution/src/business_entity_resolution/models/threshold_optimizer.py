"""Threshold optimization maximizing Macro F0.5 on validation split."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel

from ..evaluation.f05 import compute_macro_f05, evaluate_predictions
from ..utils.logger import setup_logger

logger = setup_logger("models.threshold_optimizer")


class ThresholdResult(BaseModel):
    """Result of threshold optimization search."""
    optimal_threshold: float
    best_macro_f05: float
    best_macro_precision: float
    best_macro_recall: float
    default_f05: float
    threshold_curve: List[Tuple[float, float]]


class ThresholdOptimizer:
    """Finds the decision threshold that maximizes validation F0.5."""

    def __init__(
        self,
        min_threshold: float = 0.10,
        max_threshold: float = 0.95,
        step: float = 0.01,
        default_threshold: float = 0.50,
    ):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.step = step
        self.default_threshold = default_threshold

    def optimize(
        self,
        pair_predictions: List[Tuple[str, str, float]],
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: List[str],
    ) -> ThresholdResult:
        """Searches for the threshold maximizing Macro F0.5.

        Args:
            pair_predictions: List of (s1_id, target_id, predicted_probability).
            ground_truth: Mapping of true matching target IDs per s1_id.
            all_s1_ids: Full list of Source 1 validation entities (including singletons).

        Returns:
            ThresholdResult object with optimal threshold and curve.
        """
        logger.info(f"Optimizing threshold over {len(all_s1_ids):,} entities...")

        # Group candidate probabilities by S1 entity
        candidates_by_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for s1_id, tid, prob in pair_predictions:
            candidates_by_s1[s1_id].append((tid, prob))

        thresholds = np.arange(self.min_threshold, self.max_threshold + 1e-5, self.step)
        curve: List[Tuple[float, float]] = []

        best_th = self.default_threshold
        best_score = -1.0
        default_score = 0.0

        for th in thresholds:
            th_val = round(float(th), 3)
            preds: Dict[str, Set[str]] = {}
            for s1_id in all_s1_ids:
                cands = candidates_by_s1.get(s1_id, [])
                matched = {tid for tid, prob in cands if prob >= th_val}
                preds[s1_id] = matched

            score = compute_macro_f05(ground_truth, preds, all_s1_ids)
            curve.append((th_val, round(score, 5)))

            if abs(th_val - self.default_threshold) < 0.005:
                default_score = score

            if score > best_score:
                best_score = score
                best_th = th_val

        # Re-evaluate best threshold to extract precision and recall
        best_preds: Dict[str, Set[str]] = {}
        for s1_id in all_s1_ids:
            cands = candidates_by_s1.get(s1_id, [])
            best_preds[s1_id] = {tid for tid, prob in cands if prob >= best_th}

        best_metrics = evaluate_predictions(ground_truth, best_preds, all_s1_ids)

        logger.info(
            f"Optimization complete! Optimal threshold: {best_th:.3f} "
            f"(Macro F0.5: {best_score:.5f}, Default F0.5@0.50: {default_score:.5f}, "
            f"Precision: {best_metrics.macro_precision:.5f}, Recall: {best_metrics.macro_recall:.5f})"
        )

        return ThresholdResult(
            optimal_threshold=best_th,
            best_macro_f05=round(best_score, 5),
            best_macro_precision=best_metrics.macro_precision,
            best_macro_recall=best_metrics.macro_recall,
            default_f05=round(default_score, 5),
            threshold_curve=curve,
        )
