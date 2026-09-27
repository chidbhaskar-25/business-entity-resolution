"""Inference and scoring utilities for trained entity matching models."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..data.label_builder import LabelBuilder
from ..utils.logger import setup_logger
from .matcher import BaselineMatcher

logger = setup_logger("models.predict_model")


class EntityPredictor:
    """Predictor wrapper for running inference with saved matching model artifacts."""

    def __init__(self, model_path: str = "models/baseline_model.joblib"):
        self.model_path = model_path
        self.matcher = BaselineMatcher.load(model_path)
        logger.info(
            f"EntityPredictor initialized with {self.matcher.model_type} model from {model_path} "
            f"({len(self.matcher.feature_names)} features)"
        )

    @property
    def feature_names(self) -> List[str]:
        return self.matcher.feature_names

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Computes continuous match probabilities in [0.0, 1.0] for feature rows."""
        return self.matcher.predict_proba(X)

    def predict_binary(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Computes binary match decisions for feature rows using specified decision threshold."""
        return self.matcher.predict(X, threshold=threshold)

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        X: np.ndarray,
        threshold: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """Scores candidate pairs and returns structured prediction dictionaries."""
        if len(pairs) != len(X):
            raise ValueError(f"Length mismatch: {len(pairs)} pairs vs {len(X)} feature rows.")

        probs = self.predict_probabilities(X)
        results: List[Dict[str, Any]] = []

        for (s1_id, target_id), prob in zip(pairs, probs):
            p_val = float(prob)
            results.append({
                "s1_id": s1_id,
                "target_id": target_id,
                "match_probability": round(p_val, 5),
                "is_match": 1 if p_val >= threshold else 0,
            })
        return results

    def predict_matches_map(
        self,
        pairs: List[Tuple[str, str]],
        X: np.ndarray,
        threshold: float = 0.50,
        all_s1_ids: Optional[List[str]] = None,
    ) -> Dict[str, Set[str]]:
        """Maps each Source 1 entity to its predicted matching target IDs where probability >= threshold.

        Guarantees that all requested S1 IDs (including singletons) are present in the returned dictionary.
        """
        if len(pairs) != len(X):
            raise ValueError(f"Length mismatch: {len(pairs)} pairs vs {len(X)} feature rows.")

        probs = self.predict_probabilities(X)
        matches: Dict[str, Set[str]] = defaultdict(set)

        # Pre-seed singletons / all requested S1 entities
        if all_s1_ids:
            for s1_id in all_s1_ids:
                matches[s1_id] = set()

        for (s1_id, target_id), prob in zip(pairs, probs):
            if prob >= threshold:
                matches[s1_id].add(target_id)
            elif s1_id not in matches:
                matches[s1_id] = set()

        return dict(matches)


def predict_from_dataset(
    model_path: str = "models/baseline_model.joblib",
    dataset_path: str = "outputs/train_dataset.npz",
    threshold: float = 0.50,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs inference on a saved feature dataset and optionally saves predictions."""
    predictor = EntityPredictor(model_path)
    X, y, pairs, feature_names = LabelBuilder.load_dataset_npz(dataset_path)

    probs = predictor.predict_probabilities(X)
    preds = (probs >= threshold).astype(int)

    num_pos = int(np.sum(preds))
    logger.info(
        f"Inference complete on {len(pairs):,} pairs: "
        f"{num_pos:,} positive matches ({num_pos / len(pairs):.2%}) at threshold={threshold:.2f}"
    )

    summary = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "threshold": threshold,
        "total_pairs": len(pairs),
        "predicted_positives": num_pos,
        "predicted_positive_ratio": round(num_pos / len(pairs), 5) if len(pairs) > 0 else 0.0,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved prediction summary to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using trained entity matcher")
    parser.add_argument("--model-path", type=str, default="models/baseline_model.joblib")
    parser.add_argument("--dataset-path", type=str, default="outputs/train_dataset.npz")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output", type=str, default="outputs/predictions_summary.json")
    args = parser.parse_args()

    predict_from_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        threshold=args.threshold,
        output_path=args.output,
    )
