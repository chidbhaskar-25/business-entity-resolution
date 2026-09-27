"""Model training pipeline for Business Entity Resolution baseline."""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from ..data.label_builder import LabelBuilder
from ..evaluation.f05 import evaluate_predictions
from ..utils.logger import setup_logger
from .matcher import BaselineMatcher

logger = setup_logger("models.train_model")


def train_baseline(
    dataset_path: str = "outputs/train_dataset.npz",
    output_model_path: str = "models/baseline_model.joblib",
    metrics_output_path: str = "outputs/baseline_metrics.json",
    model_type: str = "logistic_regression",
    val_size: float = 0.20,
    random_seed: int = 42,
    class_weight: str = "balanced",
) -> Dict[str, Any]:
    """Trains baseline matching model with entity-grouped validation split.

    Args:
        dataset_path: Path to pairwise training dataset .npz file.
        output_model_path: Path to save trained model artifact (.joblib).
        metrics_output_path: Path to save validation metrics JSON.
        model_type: Classifier architecture ("logistic_regression", "hist_gradient_boosting").
        val_size: Proportion of Source 1 entities reserved for validation.
        random_seed: Reproducibility seed for data split and model initialization.
        class_weight: Handling strategy for class imbalance ("balanced").

    Returns:
        Dictionary of training and validation metrics.
    """
    logger.info("=" * 70)
    logger.info("STARTING BASELINE MODEL TRAINING")
    logger.info("=" * 70)

    # 1. Load pairwise dataset
    logger.info(f"Loading training dataset from: {dataset_path}")
    X, y, pairs, feature_names = LabelBuilder.load_dataset_npz(dataset_path)
    total_pairs = len(pairs)
    total_positives = int(np.sum(y))
    logger.info(
        f"Dataset loaded: {total_pairs:,} total pairs, "
        f"{total_positives:,} positive matches ({np.mean(y):.2%}), "
        f"{total_pairs - total_positives:,} non-matches, {len(feature_names)} features"
    )

    # 2. Entity-Grouped Split (Guarantees zero entity leakage across splits)
    s1_ids = np.array([p[0] for p in pairs])
    unique_s1 = np.unique(s1_ids)
    logger.info(f"Total unique Source 1 entities in dataset: {len(unique_s1):,}")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_seed)
    train_idx, val_idx = next(gss.split(X, y, groups=s1_ids))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_s1 = set(s1_ids[train_idx])
    val_s1 = set(s1_ids[val_idx])
    leakage = train_s1 & val_s1
    assert len(leakage) == 0, f"Critical error: Entity leakage detected across train/val! ({len(leakage)} overlapping IDs)"

    logger.info(
        f"Group split (seed={random_seed}): "
        f"Train={len(train_idx):,} pairs ({len(train_s1):,} S1 entities, {np.sum(y_train):,} positives), "
        f"Validation={len(val_idx):,} pairs ({len(val_s1):,} S1 entities, {np.sum(y_val):,} positives)"
    )

    # 3. Initialize and fit baseline matcher
    matcher = BaselineMatcher(
        model_type=model_type,
        random_state=random_seed,
        class_weight=class_weight,
    )
    matcher.fit(X_train, y_train, feature_names=feature_names)

    # 4. Predict probabilities on validation set
    val_probs = matcher.predict_proba(X_val)
    val_pred = (val_probs >= 0.50).astype(int)

    # 5. Compute standard pairwise classification metrics
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_logloss = float(log_loss(y_val, val_probs))
    p_50 = float(precision_score(y_val, val_pred, zero_division=0))
    r_50 = float(recall_score(y_val, val_pred, zero_division=0))
    f1_50 = float(f1_score(y_val, val_pred, zero_division=0))
    f05_50 = float(fbeta_score(y_val, val_pred, beta=0.5, zero_division=0))

    cm = confusion_matrix(y_val, val_pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    # 6. Compute competition Macro F0.5 on validation S1 entities
    val_pairs = [pairs[i] for i in val_idx]
    gt_map: Dict[str, Set[str]] = defaultdict(set)
    pred_map: Dict[str, Set[str]] = defaultdict(set)

    for (s1_id, target_id), true_label, prob in zip(val_pairs, y_val, val_probs):
        if true_label == 1:
            gt_map[s1_id].add(target_id)
        if prob >= 0.50:
            pred_map[s1_id].add(target_id)

    # Ensure all validation S1 entities (including singletons) are in the maps
    all_val_s1_list = sorted(list(val_s1))
    er_metrics = evaluate_predictions(gt_map, pred_map, all_val_s1_list)

    # 7. Package metrics report
    metrics_summary = {
        "model_type": model_type,
        "random_seed": random_seed,
        "class_weight": class_weight,
        "total_pairs": total_pairs,
        "train_pairs": len(train_idx),
        "val_pairs": len(val_idx),
        "val_s1_entities": len(val_s1),
        "pairwise_metrics": {
            "roc_auc": round(val_roc_auc, 5),
            "average_precision_pr_auc": round(val_pr_auc, 5),
            "log_loss": round(val_logloss, 5),
            "precision_at_0_50": round(p_50, 5),
            "recall_at_0_50": round(r_50, 5),
            "f1_at_0_50": round(f1_50, 5),
            "f05_at_0_50": round(f05_50, 5),
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            },
        },
        "competition_macro_metrics_at_0_50": {
            "macro_f05": er_metrics.macro_f05,
            "macro_precision": er_metrics.macro_precision,
            "macro_recall": er_metrics.macro_recall,
            "total_entities": er_metrics.total_entities,
            "singleton_entities": er_metrics.singleton_entities,
            "singleton_accuracy": er_metrics.singleton_accuracy,
            "total_true_matches": er_metrics.total_true_matches,
            "total_predicted_matches": er_metrics.total_predicted_matches,
        },
    }

    # 8. Save trained model artifact with joblib
    matcher.metadata = metrics_summary
    matcher.save(output_model_path)

    # 9. Save metrics JSON
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved validation metrics report to: {metrics_output_path}")

    # 10. Display formatted training summary
    print("\n" + "=" * 75)
    print("BASELINE MATCHING MODEL: VALIDATION PERFORMANCE REPORT")
    print("=" * 75)
    print(f"Model Architecture:          {model_type.upper()}")
    print(f"Artifact Saved To:           {output_model_path}")
    print(f"Validation Sample:           {len(val_idx):,} pairs across {len(val_s1):,} S1 entities")
    print("-" * 75)
    print(f"Pairwise ROC AUC:            {val_roc_auc:.4f}")
    print(f"Pairwise PR AUC (Avg Prec):  {val_pr_auc:.4f}")
    print(f"Pairwise Log Loss:           {val_logloss:.4f}")
    print("-" * 75)
    print(f"Pairwise Precision (th=0.50):{p_50:.4f}")
    print(f"Pairwise Recall (th=0.50):   {r_50:.4f}")
    print(f"Pairwise F0.5 (th=0.50):     {f05_50:.4f}")
    print(f"Pairwise F1 (th=0.50):       {f1_50:.4f}")
    print("-" * 75)
    print("Pairwise Confusion Matrix:")
    print(f"  TN = {tn:6,d}  |  FP = {fp:6,d}")
    print(f"  FN = {fn:6,d}  |  TP = {tp:6,d}")
    print("-" * 75)
    print("Competition Macro Entity Resolution Metrics (Default th=0.50):")
    print(f"  Macro F0.5 Score:          {er_metrics.macro_f05:.4f}")
    print(f"  Macro Precision:           {er_metrics.macro_precision:.4f}")
    print(f"  Macro Recall:              {er_metrics.macro_recall:.4f}")
    print(f"  Singleton Accuracy:        {er_metrics.singleton_accuracy:.2%}")
    print("=" * 75 + "\n")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline entity matching model")
    parser.add_argument("--dataset-path", type=str, default="outputs/train_dataset.npz")
    parser.add_argument("--output-model", type=str, default="models/baseline_model.joblib")
    parser.add_argument("--metrics-output", type=str, default="outputs/baseline_metrics.json")
    parser.add_argument("--model-type", type=str, default="logistic_regression")
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--class-weight", type=str, default="balanced")
    args = parser.parse_args()

    train_baseline(
        dataset_path=args.dataset_path,
        output_model_path=args.output_model,
        metrics_output_path=args.metrics_output,
        model_type=args.model_type,
        val_size=args.val_size,
        random_seed=args.random_seed,
        class_weight=args.class_weight,
    )
