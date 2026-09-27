"""Entity resolution matching models using scikit-learn and joblib."""

import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..utils.logger import setup_logger

logger = setup_logger("models.matcher")


class BaselineMatcher:
    """Baseline entity matching classifier with joblib persistence and imbalance handling."""

    def __init__(
        self,
        model_type: str = "logistic_regression",
        random_state: int = 42,
        class_weight: str = "balanced",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.class_weight = class_weight
        self.config = config or {}
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.scaler: Optional[StandardScaler] = None
        self.model: Any = None
        self.optimal_threshold: float = 0.50
        self.metadata: Dict[str, Any] = {}

        self._build_estimator()

    def _build_estimator(self) -> None:
        """Constructs estimator with reproducible random seed and class imbalance handling."""
        if self.model_type in ("logistic_regression", "lr"):
            self.scaler = StandardScaler()
            c_val = float(self.config.get("C", 1.0))
            max_iter = int(self.config.get("max_iter", 1000))
            self.model = LogisticRegression(
                C=c_val,
                max_iter=max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver=self.config.get("solver", "lbfgs"),
            )
        elif self.model_type in ("hist_gradient_boosting", "gbm", "gradient_boosting"):
            self.scaler = None
            self.model = HistGradientBoostingClassifier(
                max_iter=int(self.config.get("max_iter", 100)),
                learning_rate=float(self.config.get("learning_rate", 0.1)),
                max_depth=int(self.config.get("max_depth", 8)),
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        elif self.model_type in ("random_forest", "rf"):
            self.scaler = None
            self.model = RandomForestClassifier(
                n_estimators=int(self.config.get("n_estimators", 100)),
                max_depth=int(self.config.get("max_depth", 10)),
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(
                f"Unsupported model type '{self.model_type}'. "
                f"Choose from: logistic_regression, hist_gradient_boosting, random_forest"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "BaselineMatcher":
        """Trains the model on pairwise features X and binary match labels y."""
        pos_count = int(np.sum(y))
        logger.info(
            f"Training {self.model_type} baseline on {len(X):,} candidate pairs "
            f"(positives={pos_count:,} / {np.mean(y):.2%}, negatives={len(y) - pos_count:,})..."
        )
        self.feature_names = feature_names or []

        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X)
        else:
            X_train = X

        self.model.fit(X_train, y)
        self.is_fitted = True

        self._log_feature_importance()
        return self

    def _log_feature_importance(self) -> None:
        if not self.feature_names:
            return
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            top_feats = sorted(zip(self.feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
            logger.info("Top influential features (coefficients):")
            for name, weight in top_feats[:8]:
                logger.info(f"  {name:30s}: {weight:+.4f}")
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            top_feats = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
            logger.info("Top influential features (importance):")
            for name, weight in top_feats[:8]:
                logger.info(f"  {name:30s}: {weight:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs match probability (float in [0, 1]) for each pair."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Fit or load a trained model first.")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)

        if self.scaler is not None:
            X_in = self.scaler.transform(X)
        else:
            X_in = X

        probs = self.model.predict_proba(X_in)
        return probs[:, 1].astype(np.float32)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary match decisions (1 or 0) using decision threshold."""
        th = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(np.int32)

    def save(self, filepath: str) -> None:
        """Saves model artifact to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        file_size_kb = os.path.getsize(filepath) / 1024
        logger.info(f"Saved {self.model_type} model artifact ({file_size_kb:.1f} KB) to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BaselineMatcher":
        """Loads trained model artifact from disk using joblib."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        matcher = joblib.load(filepath)
        if not isinstance(matcher, cls):
            raise TypeError(f"Loaded object is of type {type(matcher)}, expected {cls}")
        logger.info(f"Loaded {matcher.model_type} model from: {filepath} (fitted={matcher.is_fitted})")
        return matcher
