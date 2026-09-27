"""Pydantic request and response schemas for FastAPI ML service endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Health
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('healthy')")
    active_model: str = Field(..., description="Active trained model descriptor")
    model_size_bytes: int = Field(..., description="Size of active model file in bytes")
    optimal_threshold: float = Field(..., description="Currently configured optimal decision threshold")
    train_data_available: bool = Field(..., description="Whether train dataset directory exists")
    test_data_available: bool = Field(..., description="Whether test dataset directory exists")
    matching_results_exist: bool = Field(..., description="Whether matching_results.tsv has been generated")
    candidate_pairs_exist: bool = Field(..., description="Whether candidate_pairs.tsv has been generated")
    timestamp: str = Field(..., description="UTC ISO timestamp of health check")


# -----------------------------------------------------------------------------
# 2. Dataset Stats
# -----------------------------------------------------------------------------
class DatasetStatsResponse(BaseModel):
    train: Dict[str, Any] = Field(..., description="Train directory file stats and sample distributions")
    test: Dict[str, Any] = Field(..., description="Test directory file stats")
    ground_truth: Dict[str, Any] = Field(..., description="Ground truth singleton & multi-match statistics")


# -----------------------------------------------------------------------------
# 3. Train
# -----------------------------------------------------------------------------
class TrainRequest(BaseModel):
    model_type: str = Field(
        default="hist_gradient_boosting",
        description="Classifier architecture ('hist_gradient_boosting' or 'logistic_regression')",
    )
    dataset_path: str = Field(
        default="outputs/train_dataset.npz",
        description="Path to pairwise training dataset .npz",
    )
    val_size: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="Proportion of Source 1 entities reserved for validation",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    output_model_path: str = Field(
        default="models/best_model.joblib",
        description="Destination path for serialized model artifact",
    )


class TrainResponse(BaseModel):
    model_type: str
    output_model_path: str
    val_size: float
    random_seed: int
    train_pairs: int
    val_pairs: int
    runtime_seconds: float
    pairwise_metrics: Dict[str, Any]
    entity_macro_metrics: Dict[str, Any]


# -----------------------------------------------------------------------------
# 4. Evaluate
# -----------------------------------------------------------------------------
class EvaluateRequest(BaseModel):
    model_path: str = Field(
        default="models/best_model.joblib",
        description="Path to serialized model artifact to evaluate",
    )
    dataset_path: str = Field(
        default="outputs/train_dataset.npz",
        description="Path to pairwise dataset .npz",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Decision threshold (if None, loaded from best_threshold.json)",
    )
    optimize_threshold: bool = Field(
        default=False,
        description="If True, sweeps 40+ thresholds to find optimal Macro F0.5",
    )


class EvaluateResponse(BaseModel):
    model_path: str
    model_type: str
    threshold_applied: float
    macro_f05: float
    macro_precision: float
    macro_recall: float
    total_entities_evaluated: int
    singleton_entities: int
    singleton_accuracy: float
    total_true_matches: int
    total_predicted_matches: int
    threshold_optimization: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# 5. Predict
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    sample_size: Optional[int] = Field(
        default=100,
        description="Number of Source 1 entities to score (None for entire test set)",
    )
    target_sample_size: Optional[int] = Field(
        default=10000,
        description="Number of Target entities (S2/S3) to load for candidate indexing (None for all)",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Decision threshold override (defaults to selected best threshold)",
    )
    model_path: str = Field(
        default="models/best_model.joblib",
        description="Model artifact path to load",
    )
    chunk_size: int = Field(
        default=5000,
        description="Streaming chunk size for memory-bounded processing",
    )


class PredictResponse(BaseModel):
    status: str
    matching_file: str
    candidate_file: str
    sample_size: Optional[int]
    threshold_applied: float
    validation_passed: bool
    validation_errors: List[str]
    validation_stats: Dict[str, Any]


# -----------------------------------------------------------------------------
# 6. Validate
# -----------------------------------------------------------------------------
class ValidateRequest(BaseModel):
    matching_path: str = Field(
        default="output/matching_results.tsv",
        description="Path to matching_results.tsv",
    )
    candidate_path: Optional[str] = Field(
        default="output/candidate_pairs.tsv",
        description="Path to candidate_pairs.tsv",
    )
    check_target_ids: bool = Field(
        default=False,
        description="Whether to scan all 10M test targets to check ID existence",
    )
    s1_sample_size: Optional[int] = Field(
        default=None,
        description="If validating a sampled run, specify sample size",
    )


class ValidateResponse(BaseModel):
    is_valid: bool
    status: str
    criteria_checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]
    report_summary: str


# -----------------------------------------------------------------------------
# 7. Metrics
# -----------------------------------------------------------------------------
class MetricsResponse(BaseModel):
    evaluation_metrics: Optional[Dict[str, Any]] = None
    threshold_metrics: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# 8. Experiments
# -----------------------------------------------------------------------------
class ExperimentItem(BaseModel):
    experiment_id: str
    model: str
    features: str
    blocking: str
    threshold: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f05: Optional[float] = None
    candidate_recall: Optional[float] = None
    runtime_seconds: Optional[float] = None


class ExperimentsResponse(BaseModel):
    experiments: List[ExperimentItem]
    total: int


# -----------------------------------------------------------------------------
# 9. Matches Lookup
# -----------------------------------------------------------------------------
class MatchDetail(BaseModel):
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str


class CandidateDetail(BaseModel):
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str
    name_similarity: float
    address_similarity: float
    numeric_overlap: float
    composite_score: float
    country_match: bool
    model_probability: float
    is_match: bool


class EntityMatchResponse(BaseModel):
    source1_entity_id: str
    business_name: str
    business_address: str
    country: str
    is_singleton: bool
    matched_entity_ids: List[str]
    candidate_entity_ids: List[str]
    matches_details: List[MatchDetail]
    candidate_details: List[CandidateDetail] = []

