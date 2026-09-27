"""Concrete classification models for entity resolution."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseEntityMatcher
from ..utils.logger import setup_logger

logger = setup_logger("models.classifier")


class LogisticRegressionMatcher(BaseEntityMatcher):
    """Logistic Regression classifier with feature scaling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("logistic_regression", config)
        cfg = config or {}
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=cfg.get("random_state", 42),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "LogisticRegressionMatcher":
        logger.info(f"Fitting LogisticRegressionMatcher on {len(X):,} pairs (pos={np.sum(y):,})...")
        self.feature_names = feature_names or []
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True

        if self.feature_names and hasattr(self.clf, "coef_"):
            coefs = self.clf.coef_[0]
            top_feats = sorted(zip(self.feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
            logger.info("Top model coefficients:")
            for name, weight in top_feats[:8]:
                logger.info(f"  {name:30s}: {weight:+.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if len(X) == 0:
            return np.empty((0,))
        X_scaled = self.scaler.transform(X)
        probs = self.clf.predict_proba(X_scaled)
        return probs[:, 1]


class RandomForestMatcher(BaseEntityMatcher):
    """Random Forest classifier for non-linear feature interaction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("random_forest", config)
        cfg = config or {}
        self.clf = RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 10),
            random_state=cfg.get("random_state", 42),
            n_jobs=cfg.get("n_jobs", -1),
            class_weight="balanced_subsample",
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "RandomForestMatcher":
        logger.info(f"Fitting RandomForestMatcher on {len(X):,} pairs (pos={np.sum(y):,})...")
        self.feature_names = feature_names or []
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if len(X) == 0:
            return np.empty((0,))
        probs = self.clf.predict_proba(X)
        return probs[:, 1]


class HistGradientBoostingMatcher(BaseEntityMatcher):
    """Histogram-based Gradient Boosting classifier."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("hist_gradient_boosting", config)
        cfg = config or {}
        self.clf = HistGradientBoostingClassifier(
            max_iter=cfg.get("max_iter", 100),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 8),
            random_state=cfg.get("random_state", 42),
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "HistGradientBoostingMatcher":
        logger.info(f"Fitting HistGradientBoostingMatcher on {len(X):,} pairs (pos={np.sum(y):,})...")
        self.feature_names = feature_names or []
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if len(X) == 0:
            return np.empty((0,))
        probs = self.clf.predict_proba(X)
        return probs[:, 1]


def get_model(model_type: str = "logistic_regression", config: Optional[Dict[str, Any]] = None) -> BaseEntityMatcher:
    """Factory creating an entity matching model by type name."""
    clean_type = model_type.lower().replace("-", "_")
    if clean_type == "logistic_regression":
        return LogisticRegressionMatcher(config)
    elif clean_type in ("random_forest", "rf"):
        return RandomForestMatcher(config)
    elif clean_type in ("hist_gradient_boosting", "gbm", "gradient_boosting"):
        return HistGradientBoostingMatcher(config)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: logistic_regression, random_forest, hist_gradient_boosting")
