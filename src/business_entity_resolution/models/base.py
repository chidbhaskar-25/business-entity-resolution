"""Base interface for Entity Resolution classification models."""

from abc import ABC, abstractmethod
import os
import pickle
from typing import Any, Dict, List, Optional
import numpy as np


class BaseEntityMatcher(ABC):
    """Abstract base class for entity matching binary classifiers."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.50

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "BaseEntityMatcher":
        """Fits the classifier on candidate pair features X and binary match labels y."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts match probability for each candidate pair."""
        pass

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary match labels using given or optimal threshold."""
        th = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def save(self, filepath: str) -> None:
        """Serializes the model object to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "BaseEntityMatcher":
        """Loads a serialized model object from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)
