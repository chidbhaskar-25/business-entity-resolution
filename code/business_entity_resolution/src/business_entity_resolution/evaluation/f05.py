"""Exact competition-standard F0.5 evaluation metric for Business Entity Resolution."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel


class EntityMetricRecord(BaseModel):
    """Detailed evaluation metrics for an individual Source 1 entity."""
    s1_id: str
    true_count: int
    pred_count: int
    tp_count: int
    precision: float
    recall: float
    f05: float
    is_singleton: bool
    is_correct_singleton: bool


class MacroEvaluationSummary(BaseModel):
    """Macro-averaged competition evaluation report across all Source 1 entities."""
    macro_f05: float
    macro_precision: float
    macro_recall: float
    total_entities: int
    singleton_entities: int
    singleton_accuracy: float
    non_singleton_entities: int
    non_singleton_macro_f05: float
    total_true_matches: int
    total_predicted_matches: int
    total_true_positives: int
    micro_precision: float
    micro_recall: float


def compute_entity_f05(
    true_set: Set[str],
    pred_set: Set[str],
) -> Tuple[float, float, float]:
    """Computes F0.5, Precision, and Recall for a single Source 1 entity.

    Formula & Competition Edge Cases:
    --------------------------------
    F_beta = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall)
    For beta = 0.5:
    F0.5 = (1.25 * P * R) / (0.25 * P + R)

    Edge Case Rules:
    1. Singleton with No Predictions (T = {}, P = {}):
       Entity represents a real-world singleton correctly predicted with no links.
       Precision = 1.0, Recall = 1.0, F0.5 = 1.0.
    2. Singleton with False Predictions (T = {}, P != {}):
       Entity is a singleton, but false positive links were predicted.
       Precision = 0.0, Recall = 1.0, F0.5 = 0.0.
    3. Non-Singleton with Empty Predictions (T != {}, P = {}):
       Matches were missed entirely.
       Precision = 1.0, Recall = 0.0, F0.5 = 0.0.
    4. Non-Singleton with Matches (T != {}, P != {}):
       TP = |T & P|
       Precision = TP / |P|
       Recall = TP / |T|
       F0.5 = (1.25 * P * R) / (0.25 * P + R) if (0.25 * P + R) > 0 else 0.0

    Args:
        true_set: Set of true matching target record IDs for this S1 entity.
        pred_set: Set of predicted matching target record IDs for this S1 entity.

    Returns:
        Tuple of (f05: float, precision: float, recall: float).
    """
    # Rule 1: Correctly identified singleton
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0

    # Rule 2: False links on singleton
    if not true_set and pred_set:
        return 0.0, 0.0, 1.0

    # Rule 3: Non-singleton predicted as empty
    if true_set and not pred_set:
        return 0.0, 1.0, 0.0

    # Rule 4: Both non-empty
    tp = len(true_set & pred_set)
    p = tp / len(pred_set)
    r = tp / len(true_set)

    denominator = 0.25 * p + r
    if denominator > 0.0:
        f05 = (1.25 * p * r) / denominator
    else:
        f05 = 0.0

    return float(f05), float(p), float(r)


def evaluate_entity_records(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    all_s1_ids: Optional[List[str]] = None,
) -> Tuple[MacroEvaluationSummary, List[EntityMetricRecord]]:
    """Evaluates entity-level matching across all Source 1 entities.

    Args:
        ground_truth: Mapping of s1_id -> set of true matching target IDs.
        predictions: Mapping of s1_id -> set of predicted matching target IDs.
        all_s1_ids: Comprehensive list of Source 1 entity IDs (including singletons).
                    Defaults to the union of keys in ground_truth and predictions.

    Returns:
        Tuple of (MacroEvaluationSummary, List[EntityMetricRecord]).
    """
    if all_s1_ids is not None:
        eval_ids = list(all_s1_ids)
    else:
        eval_ids = sorted(list(set(ground_truth.keys()) | set(predictions.keys())))

    if not eval_ids:
        raise ValueError("No Source 1 entities provided for evaluation.")

    records: List[EntityMetricRecord] = []
    f05_list: List[float] = []
    p_list: List[float] = []
    r_list: List[float] = []

    singleton_count = 0
    correct_singleton_count = 0
    non_singleton_f05_list: List[float] = []

    total_true = 0
    total_pred = 0
    total_tp = 0

    for s1_id in eval_ids:
        t_set = ground_truth.get(s1_id, set())
        p_set = predictions.get(s1_id, set())

        f05, p, r = compute_entity_f05(t_set, p_set)
        tp = len(t_set & p_set)

        is_singleton = len(t_set) == 0
        is_correct_singleton = is_singleton and len(p_set) == 0

        if is_singleton:
            singleton_count += 1
            if is_correct_singleton:
                correct_singleton_count += 1
        else:
            non_singleton_f05_list.append(f05)

        total_true += len(t_set)
        total_pred += len(p_set)
        total_tp += tp

        f05_list.append(f05)
        p_list.append(p)
        r_list.append(r)

        records.append(
            EntityMetricRecord(
                s1_id=s1_id,
                true_count=len(t_set),
                pred_count=len(p_set),
                tp_count=tp,
                precision=round(p, 5),
                recall=round(r, 5),
                f05=round(f05, 5),
                is_singleton=is_singleton,
                is_correct_singleton=is_correct_singleton,
            )
        )

    macro_f05 = float(np.mean(f05_list))
    macro_p = float(np.mean(p_list))
    macro_r = float(np.mean(r_list))

    singleton_acc = (correct_singleton_count / singleton_count) if singleton_count > 0 else 1.0
    non_singleton_macro_f05 = float(np.mean(non_singleton_f05_list)) if non_singleton_f05_list else 0.0

    micro_p = (total_tp / total_pred) if total_pred > 0 else 0.0
    micro_r = (total_tp / total_true) if total_true > 0 else 0.0

    summary = MacroEvaluationSummary(
        macro_f05=round(macro_f05, 5),
        macro_precision=round(macro_p, 5),
        macro_recall=round(macro_r, 5),
        total_entities=len(eval_ids),
        singleton_entities=singleton_count,
        singleton_accuracy=round(singleton_acc, 5),
        non_singleton_entities=len(non_singleton_f05_list),
        non_singleton_macro_f05=round(non_singleton_macro_f05, 5),
        total_true_matches=total_true,
        total_predicted_matches=total_pred,
        total_true_positives=total_tp,
        micro_precision=round(micro_p, 5),
        micro_recall=round(micro_r, 5),
    )

    return summary, records


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    all_s1_ids: Optional[List[str]] = None,
) -> float:
    """Convenience helper returning purely the competition Macro F0.5 score."""
    summary, _ = evaluate_entity_records(ground_truth, predictions, all_s1_ids)
    return summary.macro_f05


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    all_s1_ids: Optional[List[str]] = None,
) -> MacroEvaluationSummary:
    """Convenience helper returning MacroEvaluationSummary across entities."""
    summary, _ = evaluate_entity_records(ground_truth, predictions, all_s1_ids)
    return summary


# Backward compatibility alias
EntityResolutionMetrics = MacroEvaluationSummary

