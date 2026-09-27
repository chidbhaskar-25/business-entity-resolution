"""Domain service orchestrating ML pipeline business logic for backend API."""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ..data.label_builder import LabelBuilder
from ..data.loader import DataLoader
from ..data.submission_validator import SubmissionValidationResult, SubmissionValidator
from ..data.validator import OutputValidator
from ..evaluation.threshold_optimizer import ThresholdOptimizer
from ..features.feature_builder import FeatureBuilder
from ..inference.output_writer import SubmissionWriter
from ..inference.pipeline import InferencePipeline
from ..models.matcher import BaselineMatcher
from ..models.train_model import train_baseline
from ..utils.logger import setup_logger

logger = setup_logger("services.ml_service")


class MLService:
    """Core domain service orchestrating training, evaluation, inference, and data inspection."""

    def __init__(
        self,
        base_dir: str = ".",
        data_train_dir: str = "data/train",
        data_test_dir: str = "data/test",
        models_dir: str = "models",
        outputs_dir: str = "outputs",
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.data_train_dir = os.path.join(self.base_dir, data_train_dir)
        self.data_test_dir = os.path.join(self.base_dir, data_test_dir)
        self.models_dir = os.path.join(self.base_dir, models_dir)
        self.outputs_dir = os.path.join(self.base_dir, outputs_dir)

        # In-memory caches for fast entity lookups
        self._s1_cache: Dict[str, Dict[str, str]] = {}
        self._target_cache: Dict[str, Dict[str, str]] = {}

    # -------------------------------------------------------------------------
    # 1. Health & Environment Status
    # -------------------------------------------------------------------------
    def get_health_status(self) -> Dict[str, Any]:
        """Returns system health, available model artifacts, and environment status."""
        best_model_path = os.path.join(self.models_dir, "best_model.joblib")
        baseline_model_path = os.path.join(self.models_dir, "baseline_model.joblib")
        best_threshold_path = os.path.join(self.models_dir, "best_threshold.json")

        active_model = "None"
        model_size_bytes = 0
        if os.path.exists(best_model_path):
            active_model = "hist_gradient_boosting (best_model.joblib)"
            model_size_bytes = os.path.getsize(best_model_path)
        elif os.path.exists(baseline_model_path):
            active_model = "logistic_regression (baseline_model.joblib)"
            model_size_bytes = os.path.getsize(baseline_model_path)

        current_threshold = 0.50
        if os.path.exists(best_threshold_path):
            try:
                with open(best_threshold_path, "r", encoding="utf-8") as f:
                    th_data = json.load(f)
                    current_threshold = float(th_data.get("optimal_threshold", 0.50))
            except Exception as e:
                logger.warning(f"Failed to read threshold: {e}")

        # Check outputs
        matching_out = os.path.join(self.outputs_dir, "matching_results.tsv")
        candidate_out = os.path.join(self.outputs_dir, "candidate_pairs.tsv")

        return {
            "status": "healthy",
            "active_model": active_model,
            "model_size_bytes": model_size_bytes,
            "optimal_threshold": current_threshold,
            "train_data_available": os.path.exists(self.data_train_dir),
            "test_data_available": os.path.exists(self.data_test_dir),
            "matching_results_exist": os.path.exists(matching_out),
            "candidate_pairs_exist": os.path.exists(candidate_out),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # -------------------------------------------------------------------------
    # 2. Dataset Statistics
    # -------------------------------------------------------------------------
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Extracts dataset statistics for train and test directories."""
        stats: Dict[str, Any] = {
            "train": {},
            "test": {},
            "ground_truth": {},
        }

        # Train data summary
        if os.path.exists(self.data_train_dir):
            loader = DataLoader(self.data_train_dir)
            files = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]
            for f in files:
                f_path = os.path.join(self.data_train_dir, f)
                if os.path.exists(f_path):
                    size_mb = round(os.path.getsize(f_path) / (1024 * 1024), 2)
                    stats["train"][f] = {
                        "size_mb": size_mb,
                        "exists": True,
                    }

            # Inspect sample for country breakdown and missing values
            try:
                s1_sample = loader.load_source_df("train_source1.tsv", nrows=5000)
                stats["train"]["source1_sample_count"] = len(s1_sample)
                stats["train"]["source1_countries"] = s1_sample["country"].value_counts().to_dict()
                stats["train"]["source1_missing_address_pct"] = round(
                    float(s1_sample["business_address"].eq("").mean() * 100), 2
                )
            except Exception as e:
                logger.warning(f"Error inspecting train sample: {e}")

            # Ground truth summary
            gt_path = os.path.join(self.data_train_dir, "train_ground_truth.tsv")
            if os.path.exists(gt_path):
                try:
                    gt_map = loader.load_ground_truth("train_ground_truth.tsv", nrows=10000)
                    total_entities = len(gt_map)
                    singletons = sum(1 for m in gt_map.values() if len(m) == 0)
                    multi_matches = sum(1 for m in gt_map.values() if len(m) > 1)
                    stats["ground_truth"] = {
                        "sampled_entities": total_entities,
                        "singleton_count": singletons,
                        "singleton_percentage": round(singletons / max(1, total_entities) * 100, 2),
                        "multi_match_count": multi_matches,
                        "multi_match_percentage": round(multi_matches / max(1, total_entities) * 100, 2),
                    }
                except Exception as e:
                    logger.warning(f"Error inspecting ground truth: {e}")

        # Test data summary
        if os.path.exists(self.data_test_dir):
            for f in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
                f_path = os.path.join(self.data_test_dir, f)
                if os.path.exists(f_path):
                    size_mb = round(os.path.getsize(f_path) / (1024 * 1024), 2)
                    stats["test"][f] = {
                        "size_mb": size_mb,
                        "exists": True,
                    }

        return stats

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    def train_model(
        self,
        model_type: str = "hist_gradient_boosting",
        dataset_path: str = "outputs/train_dataset.npz",
        val_size: float = 0.20,
        random_seed: int = 42,
        output_model_path: str = "models/best_model.joblib",
    ) -> Dict[str, Any]:
        """Trains matching model and saves artifact."""
        full_dataset_path = os.path.join(self.base_dir, dataset_path)
        full_output_model_path = os.path.join(self.base_dir, output_model_path)
        metrics_out = os.path.join(self.outputs_dir, "training_metrics.json")

        if not os.path.exists(full_dataset_path):
            raise FileNotFoundError(f"Training dataset not found at {full_dataset_path}. Build dataset first.")

        t0 = time.time()
        res = train_baseline(
            dataset_path=full_dataset_path,
            output_model_path=full_output_model_path,
            metrics_output_path=metrics_out,
            model_type=model_type,
            val_size=val_size,
            random_seed=random_seed,
        )
        elapsed = round(time.time() - t0, 2)
        res["runtime_seconds"] = elapsed
        res["output_model_path"] = full_output_model_path
        res["val_size"] = val_size
        res["random_seed"] = random_seed
        res["entity_macro_metrics"] = res.get("competition_macro_metrics_at_0_50", {})
        return res

    # -------------------------------------------------------------------------
    # 4. Evaluation & Threshold Optimization
    # -------------------------------------------------------------------------
    def evaluate_model(
        self,
        model_path: str = "models/best_model.joblib",
        dataset_path: str = "outputs/train_dataset.npz",
        threshold: Optional[float] = None,
        optimize_threshold: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates model performance and optionally optimizes threshold for Macro F0.5."""
        full_model_path = os.path.join(self.base_dir, model_path)
        full_dataset_path = os.path.join(self.base_dir, dataset_path)

        if not os.path.exists(full_model_path):
            full_model_path = os.path.join(self.base_dir, "models/baseline_model.joblib")
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(f"No trained model artifact found at {model_path}")

        matcher = BaselineMatcher.load(full_model_path)
        X, y, pairs, feature_names = LabelBuilder.load_dataset_npz(full_dataset_path)

        probs = matcher.predict_proba(X)

        # Default threshold
        eval_threshold = threshold
        if eval_threshold is None:
            best_th_path = os.path.join(self.models_dir, "best_threshold.json")
            if os.path.exists(best_th_path):
                with open(best_th_path, "r", encoding="utf-8") as f:
                    eval_threshold = float(json.load(f).get("optimal_threshold", 0.50))
            else:
                eval_threshold = 0.50

        # Optimizer sweep if requested
        optimizer_results = None
        optimal_th = eval_threshold
        if optimize_threshold:
            optimizer = ThresholdOptimizer(min_threshold=0.10, max_threshold=0.95, step=0.02)
            opt_res = optimizer.find_optimal_threshold(pairs=pairs, y_true=y, y_probs=probs)
            optimizer_results = {
                "optimal_threshold": opt_res.optimal_threshold,
                "best_macro_f05": opt_res.best_macro_f05,
                "best_macro_precision": opt_res.best_macro_precision,
                "best_macro_recall": opt_res.best_macro_recall,
            }
            optimal_th = opt_res.optimal_threshold

        # Macro F0.5 calculation at eval_threshold
        pred_map: Dict[str, Set[str]] = {}
        truth_map: Dict[str, Set[str]] = {}
        for (s1_id, t_id), true_label, prob in zip(pairs, y, probs):
            if s1_id not in truth_map:
                truth_map[s1_id] = set()
                pred_map[s1_id] = set()
            if true_label == 1:
                truth_map[s1_id].add(t_id)
            if prob >= eval_threshold:
                pred_map[s1_id].add(t_id)

        from ..evaluation.f05 import evaluate_predictions
        eval_metrics = evaluate_predictions(truth_map, pred_map)

        return {
            "model_path": full_model_path,
            "model_type": matcher.model_type,
            "threshold_applied": eval_threshold,
            "macro_f05": eval_metrics.macro_f05,
            "macro_precision": eval_metrics.macro_precision,
            "macro_recall": eval_metrics.macro_recall,
            "total_entities_evaluated": eval_metrics.total_entities,
            "singleton_entities": eval_metrics.singleton_entities,
            "singleton_accuracy": eval_metrics.singleton_accuracy,
            "total_true_matches": eval_metrics.total_true_matches,
            "total_predicted_matches": eval_metrics.total_predicted_matches,
            "threshold_optimization": optimizer_results,
        }

    # -------------------------------------------------------------------------
    # 5. Prediction / Inference Pipeline
    # -------------------------------------------------------------------------
    def run_prediction(
        self,
        sample_size: Optional[int] = 100,
        target_sample_size: Optional[int] = 10000,
        threshold: Optional[float] = None,
        model_path: str = "models/best_model.joblib",
        chunk_size: int = 5000,
    ) -> Dict[str, Any]:
        """Runs inference pipeline over test records and validates output compliance."""
        matching_out = os.path.join(self.outputs_dir, "matching_results.tsv")
        candidate_out = os.path.join(self.outputs_dir, "candidate_pairs.tsv")

        pipeline = InferencePipeline(
            model_path=model_path,
            threshold=threshold,
        )

        val_result = pipeline.run_full_inference(
            test_dir=self.data_test_dir,
            train_dir=self.data_train_dir,
            matching_output_path=matching_out,
            candidate_output_path=candidate_out,
            sample_size=sample_size,
            target_sample_size=target_sample_size,
            chunk_size=chunk_size,
        )

        return {
            "status": "success",
            "matching_file": matching_out,
            "candidate_file": candidate_out,
            "sample_size": sample_size,
            "threshold_applied": pipeline.threshold,
            "validation_passed": val_result.is_valid,
            "validation_errors": val_result.errors,
            "validation_stats": val_result.stats,
        }

    # -------------------------------------------------------------------------
    # 6. Submission Validation
    # -------------------------------------------------------------------------
    def validate_submission(
        self,
        matching_path: str = "outputs/matching_results.tsv",
        candidate_path: Optional[str] = "outputs/candidate_pairs.tsv",
        check_target_ids: bool = False,
        s1_sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes strict submission validation across all 11 criteria."""
        full_match = os.path.join(self.base_dir, matching_path)
        full_cand = os.path.join(self.base_dir, candidate_path) if candidate_path else None

        expected_s1 = None
        if s1_sample_size is not None and os.path.isdir(self.data_test_dir):
            s1_file = os.path.join(self.data_test_dir, "test_source1.tsv")
            if os.path.isfile(s1_file):
                expected_s1 = set()
                with open(s1_file, "r", encoding="utf-8") as f:
                    next(f)
                    for _ in range(s1_sample_size):
                        line = f.readline()
                        if not line:
                            break
                        expected_s1.add(line.split("\t", 1)[0].strip())

        result = SubmissionValidator.validate_submission(
            matching_path=full_match,
            candidate_path=full_cand,
            test_dir=self.data_test_dir,
            expected_s1_ids=expected_s1,
            check_target_ids=check_target_ids,
        )

        return {
            "is_valid": result.is_valid,
            "status": "PASS (100% Compliant)" if result.is_valid else "FAIL",
            "criteria_checks": result.criteria_checks,
            "errors": result.errors,
            "warnings": result.warnings,
            "stats": result.stats,
            "report_summary": result.report(),
        }

    # -------------------------------------------------------------------------
    # 7. Metrics & Experiments Retrieval
    # -------------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """Retrieves stored baseline evaluation metrics and threshold curve data."""
        metrics_file = os.path.join(self.outputs_dir, "metrics.json")
        th_file = os.path.join(self.models_dir, "best_threshold.json")

        output: Dict[str, Any] = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                output["evaluation_metrics"] = json.load(f)
        if os.path.exists(th_file):
            with open(th_file, "r", encoding="utf-8") as f:
                output["threshold_metrics"] = json.load(f)

        return output

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Retrieves controlled experiments log (Experiments A through F)."""
        exp_file = os.path.join(self.outputs_dir, "experiments.csv")
        if not os.path.exists(exp_file):
            return []

        experiments = []
        with open(exp_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiments.append({
                    "experiment_id": row.get("experiment_id", ""),
                    "model": row.get("model", ""),
                    "features": row.get("features", ""),
                    "blocking": row.get("blocking", ""),
                    "threshold": float(row["threshold"]) if row.get("threshold") else None,
                    "precision": float(row["precision"]) if row.get("precision") else None,
                    "recall": float(row["recall"]) if row.get("recall") else None,
                    "f05": float(row["F0.5"]) if row.get("F0.5") else None,
                    "candidate_recall": float(row["candidate_recall"]) if row.get("candidate_recall") else None,
                    "runtime_seconds": float(row["runtime"]) if row.get("runtime") else None,
                })
        return experiments

    # -------------------------------------------------------------------------
    # 8. Entity Match Lookup
    # -------------------------------------------------------------------------
    def get_matches_for_entity(self, source1_entity_id: str) -> Dict[str, Any]:
        """Looks up a Source 1 entity and returns its record, matches, candidates, and similarity details."""
        sid = source1_entity_id.strip()
        if not sid.startswith("S1-"):
            raise ValueError(f"Invalid Source 1 entity ID format: '{sid}'. Expected prefix 'S1-'.")

        # 1. Look up entity record in test or train S1
        s1_record = None
        for s1_file in [
            os.path.join(self.data_test_dir, "test_source1.tsv"),
            os.path.join(self.data_train_dir, "train_source1.tsv"),
        ]:
            if os.path.exists(s1_file):
                with open(s1_file, "r", encoding="utf-8") as f:
                    header = [c.strip().lower() for c in f.readline().rstrip("\r\n").split("\t")]
                    for line in f:
                        if line.startswith(sid + "\t"):
                            parts = [p.strip() for p in line.rstrip("\r\n").split("\t")]
                            s1_record = dict(zip(header, parts))
                            break
            if s1_record:
                break

        if not s1_record:
            raise KeyError(f"Entity '{sid}' not found in Source 1 dataset.")

        # 2. Look up matches from matching_results.tsv or ground truth
        matched_ids: List[str] = []
        matching_tsv = os.path.join(self.outputs_dir, "matching_results.tsv")
        if os.path.exists(matching_tsv):
            with open(matching_tsv, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(sid + "\t"):
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 2 and parts[1].strip():
                            matched_ids = [m.strip() for m in parts[1].split(",") if m.strip()]
                        break

        # 3. Look up candidate target IDs
        candidate_ids: List[str] = []
        cand_tsv = os.path.join(self.outputs_dir, "candidate_pairs.tsv")
        if os.path.exists(cand_tsv):
            with open(cand_tsv, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(sid + "\t"):
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 2 and parts[1].strip():
                            candidate_ids = [c.strip() for c in parts[1].split(",") if c.strip()]
                        break

        # 4. Resolve details for matched target IDs
        matched_details: List[Dict[str, Any]] = []
        for tid in matched_ids:
            target_record = self._lookup_target_record(tid)
            matched_details.append({
                "entity_id": tid,
                "business_name": target_record.get("business_name", ""),
                "business_address": target_record.get("business_address", ""),
                "country": target_record.get("country", ""),
                "source": "Source 2" if tid.startswith("S2-") else "Source 3",
            })

        # 5. Resolve candidate records, similarity features, model probabilities, and final decision
        candidate_details: List[Dict[str, Any]] = []
        feature_builder = FeatureBuilder()
        matcher = None
        best_model_path = os.path.join(self.models_dir, "best_model.joblib")
        if os.path.exists(best_model_path):
            try:
                matcher = BaselineMatcher.load(best_model_path)
            except Exception:
                pass

        for tid in candidate_ids:
            target_rec = self._lookup_target_record(tid)
            feats = feature_builder.extract_pair_dict(s1_record, target_rec)
            prob = 0.0
            if matcher is not None:
                try:
                    vec = np.array([[feats.get(fn, 0.0) for fn in feature_builder.feature_names]], dtype=np.float32)
                    prob = float(matcher.predict_proba(vec)[0])
                except Exception:
                    prob = float(feats.get("composite_heuristic_score", 0.0))
            else:
                prob = float(feats.get("composite_heuristic_score", 0.0))

            candidate_details.append({
                "entity_id": tid,
                "business_name": target_rec.get("business_name", ""),
                "business_address": target_rec.get("business_address", ""),
                "country": target_rec.get("country", ""),
                "source": "Source 2" if tid.startswith("S2-") else "Source 3",
                "name_similarity": round(float(feats.get("name_token_sort_ratio", feats.get("name_levenshtein_sim", 0.0))), 3),
                "address_similarity": round(float(feats.get("addr_token_sort_ratio", feats.get("addr_levenshtein_sim", 0.0))), 3),
                "numeric_overlap": round(float(feats.get("addr_numeric_jaccard", 0.0)), 3),
                "composite_score": round(float(feats.get("composite_heuristic_score", 0.0)), 3),
                "country_match": bool(float(feats.get("country_exact_match", 1.0)) == 1.0),
                "model_probability": round(prob, 4),
                "is_match": tid in matched_ids,
            })

        return {
            "source1_entity_id": sid,
            "business_name": s1_record.get("business_name", ""),
            "business_address": s1_record.get("business_address", ""),
            "country": s1_record.get("country", ""),
            "is_singleton": len(matched_ids) == 0,
            "matched_entity_ids": matched_ids,
            "candidate_entity_ids": candidate_ids,
            "matches_details": matched_details,
            "candidate_details": candidate_details,
        }

    def _lookup_target_record(self, target_id: str) -> Dict[str, str]:
        """Scans test/train S2 and S3 for target entity record."""
        if target_id in self._target_cache:
            return self._target_cache[target_id]

        target_file = "test_source2.tsv" if target_id.startswith("S2-") else "test_source3.tsv"
        for d in [self.data_test_dir, self.data_train_dir]:
            f_path = os.path.join(d, target_file)
            if os.path.exists(f_path):
                with open(f_path, "r", encoding="utf-8") as f:
                    header = [c.strip().lower() for c in f.readline().rstrip("\r\n").split("\t")]
                    for line in f:
                        if line.startswith(target_id + "\t"):
                            parts = [p.strip() for p in line.rstrip("\r\n").split("\t")]
                            rec = dict(zip(header, parts))
                            self._target_cache[target_id] = rec
                            return rec

        return {"entity_id": target_id}
