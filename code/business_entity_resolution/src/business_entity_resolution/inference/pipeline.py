"""Production inference pipeline generating competition matching_results.tsv and candidate_pairs.tsv."""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import yaml

from ..blocking.candidate_generator import CandidateGenerator
from ..data.loader import DataLoader
from ..data.submission_validator import SubmissionValidationResult, SubmissionValidator
from ..data.validator import OutputValidator, ValidationReport
from ..features.feature_builder import FeatureBuilder
from ..models.matcher import BaselineMatcher
from .output_writer import SubmissionWriter
from ..utils.logger import setup_logger

logger = setup_logger("inference.pipeline")


class InferencePipeline:
    """End-to-end inference engine for Business Entity Resolution test scoring."""

    def __init__(
        self,
        model: Optional[BaselineMatcher] = None,
        model_path: str = "models/best_model.joblib",
        threshold_path: str = "models/best_threshold.json",
        feature_builder: Optional[FeatureBuilder] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        max_candidates_per_entity: int = 25,
        threshold: Optional[float] = None,
    ):
        # 1. Load trained model
        if model is not None:
            self.model = model
        else:
            if not os.path.exists(model_path):
                # Fallback to baseline if best_model is not available
                fallback_path = "models/baseline_model.joblib"
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                else:
                    raise FileNotFoundError(f"Model artifact not found at {model_path} or {fallback_path}")
            self.model = BaselineMatcher.load(model_path)

        # 2. Determine decision threshold
        if threshold is not None:
            self.threshold = float(threshold)
        elif os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                th_data = json.load(f)
                self.threshold = float(th_data.get("optimal_threshold", 0.50))
        elif os.path.exists("models/threshold.json"):
            with open("models/threshold.json", "r", encoding="utf-8") as f:
                th_data = json.load(f)
                self.threshold = float(th_data.get("optimal_threshold", 0.50))
        else:
            self.threshold = 0.50

        # 3. Setup candidate generator and feature builder
        self.candidate_generator = candidate_generator or CandidateGenerator(
            max_candidates_per_entity=max_candidates_per_entity
        )
        self.feature_builder = feature_builder or FeatureBuilder()

        logger.info(
            f"InferencePipeline initialized with {self.model.model_type} model "
            f"(threshold={self.threshold:.3f}, max_cands={self.candidate_generator.max_candidates_per_entity})"
        )

    def initialize_feature_builder(self, train_dir: str = "data/train", reference_nrows: int = 10000) -> None:
        """Fits TF-IDF on training reference data to strictly prevent test data leakage."""
        if (
            self.feature_builder.name_extractor.tfidf_vectorizer is None
            or self.feature_builder.address_extractor.tfidf_vectorizer is None
        ):
            logger.info(f"Fitting FeatureBuilder TF-IDF vectorizers on {reference_nrows:,} training reference entities...")
            loader = DataLoader(train_dir)
            train_s1 = loader.load_source_df("train_source1.tsv", nrows=reference_nrows)
            self.feature_builder.fit_reference_tfidf(
                s1_names=train_s1["business_name"].tolist(),
                s1_addresses=train_s1["business_address"].tolist(),
            )

    def run_inference_on_batch(
        self,
        s1_df: pd.DataFrame,
        target_lookup: Dict[str, Dict[str, str]],
        candidate_map: Dict[str, List[str]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Scores candidate pairs for a batch of Source 1 entities.

        Args:
            s1_df: DataFrame of Source 1 entities in this batch.
            target_lookup: Dictionary of target records indexed by entity_id.
            candidate_map: Dictionary mapping s1_id -> list of candidate target_ids.

        Returns:
            Tuple of (matches_by_s1, candidates_by_s1).
        """
        s1_lookup = {r["entity_id"]: r for r in s1_df.to_dict(orient="records")}

        # Flatten candidate pairs
        candidate_pairs: List[Tuple[str, str]] = []
        for s1_id, cand_ids in candidate_map.items():
            for target_id in cand_ids:
                if target_id in target_lookup:
                    candidate_pairs.append((s1_id, target_id))

        matches_by_s1: Dict[str, List[str]] = {sid: [] for sid in s1_lookup}
        cands_by_s1: Dict[str, List[str]] = {sid: candidate_map.get(sid, []) for sid in s1_lookup}

        if not candidate_pairs:
            return matches_by_s1, cands_by_s1

        # Extract features
        X, valid_pairs = self.feature_builder.build_feature_matrix(
            candidate_pairs=candidate_pairs,
            s1_lookup=s1_lookup,
            target_lookup=target_lookup,
        )

        if len(X) == 0:
            return matches_by_s1, cands_by_s1

        # Score probabilities
        probs = self.model.predict_proba(X)

        # Apply threshold and populate matches
        for (sid, tid), prob in zip(valid_pairs, probs):
            if prob >= self.threshold:
                if tid not in matches_by_s1[sid]:
                    matches_by_s1[sid].append(tid)

        return matches_by_s1, cands_by_s1

    def run_full_inference(
        self,
        test_dir: str = "data/test",
        train_dir: str = "data/train",
        matching_output_path: str = "outputs/matching_results.tsv",
        candidate_output_path: str = "outputs/candidate_pairs.tsv",
        sample_size: Optional[int] = None,
        target_sample_size: Optional[int] = None,
        chunk_size: int = 5000,
    ) -> ValidationReport:
        """Executes full inference pipeline over test records and writes validated TSVs.

        Rules Enforced:
        1. Exact TSV formatting with required headers:
           - matching_results.tsv: source1_entity_id\tmatched_entity_ids
           - candidate_pairs.tsv: source1_entity_id\tcandidate_entity_ids
        2. Exactly one row per test Source 1 entity.
        3. Comma-separated match lists with NO duplicate IDs.
        4. Support for zero matches (singletons preserved as empty match list).
        5. Support for one or multiple matches.
        6. Matches must be a strict subset of candidate pairs.
        7. Only S2-/S3- target IDs allowed (zero self-matches).

        Args:
            test_dir: Path to directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv.
            train_dir: Path to train directory for fitting TF-IDF reference models.
            matching_output_path: Destination path for matching_results.tsv.
            candidate_output_path: Destination path for candidate_pairs.tsv.
            sample_size: Optional row limit for test Source 1 entities (None for full set).
            target_sample_size: Optional target sample limit (None for all targets).
            chunk_size: S1 batch size for streaming processing.

        Returns:
            ValidationReport detailing submission compliance.
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPETITION INFERENCE PIPELINE")
        logger.info("=" * 80)
        t_start = time.time()

        os.makedirs(os.path.dirname(matching_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(candidate_output_path), exist_ok=True)

        loader = DataLoader(test_dir)

        # 1. Initialize feature builder on training reference data (NO LEAKAGE)
        self.initialize_feature_builder(train_dir=train_dir)

        # 2. Load Source 1 test records
        logger.info(f"Loading test Source 1 entities (sample_size={sample_size})...")
        s1_df = loader.load_source_df("test_source1.tsv", nrows=sample_size)
        total_s1 = len(s1_df)
        logger.info(f"Loaded {total_s1:,} Source 1 test records.")

        # Identify unique countries (Open-Set support: e.g. France, US, India)
        unique_countries = sorted([c for c in s1_df["country"].unique() if c])
        logger.info(f"Open-set country partition detection: {unique_countries}")

        # 3. Load and Index Target records (Source 2 and Source 3)
        # Note: In entity resolution, candidates never match across countries.
        logger.info("Loading test Target records (S2 and S3)...")
        s2_df = loader.load_source_df("test_source2.tsv", nrows=target_sample_size)
        s3_df = loader.load_source_df("test_source3.tsv", nrows=target_sample_size)
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)
        del s2_df, s3_df

        logger.info(f"Loaded {len(target_df):,} test target records. Indexing into country partitions...")
        self.candidate_generator.fit_target_records(target_df)

        target_lookup = {r["entity_id"]: r for r in target_df.to_dict(orient="records")}
        del target_df
        logger.info(f"Target record lookup populated with {len(target_lookup):,} entities.")

        # 4. Process S1 entities in streaming chunks and write output TSVs
        logger.info(f"Scoring entities in chunks of {chunk_size:,} with threshold={self.threshold:.3f}...")

        with SubmissionWriter(
            matching_path=matching_output_path,
            candidate_path=candidate_output_path,
            enforce_subset=True,
        ) as writer:
            num_chunks = (total_s1 + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                t_chunk_0 = time.time()
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_s1)
                chunk_s1 = s1_df.iloc[start_idx:end_idx]

                # Step 5: Candidate generation
                cand_map, _ = self.candidate_generator.generate_candidates_batch(chunk_s1)

                # Step 6, 7, 8, 9: Feature extraction, scoring, thresholding, match list creation
                chunk_matches, chunk_cands = self.run_inference_on_batch(
                    s1_df=chunk_s1,
                    target_lookup=target_lookup,
                    candidate_map=cand_map,
                )

                # Step 10 & 11: Write line-by-line via SubmissionWriter
                for _, row in chunk_s1.iterrows():
                    sid = str(row["entity_id"]).strip()
                    m_list = chunk_matches.get(sid, [])
                    c_list = chunk_cands.get(sid, [])
                    writer.write_entity_result(sid, m_list, c_list)

                chunk_elapsed = time.time() - t_chunk_0
                logger.info(
                    f"Processed chunk {i+1}/{num_chunks} ({end_idx:,}/{total_s1:,} S1 entities) in {chunk_elapsed:.2f}s. "
                    f"Matches so far: {writer.stats['total_matches']:,} (singletons: {writer.stats['singleton_count']:,})"
                )

        total_elapsed = time.time() - t_start
        logger.info(
            f"Inference run complete in {total_elapsed:.2f}s! Generated:\n"
            f"  {matching_output_path}: {writer.stats['total_s1_written']:,} rows ({writer.stats['singleton_count']:,} singletons, {writer.stats['multi_match_count']:,} multi-matches)\n"
            f"  {candidate_output_path}: {writer.stats['total_s1_written']:,} rows ({writer.stats['total_candidates']:,} candidate pairs)"
        )

        # 5. Output Validation Check across all 11 criteria
        logger.info("Executing strict submission validation checks across all 11 criteria...")
        required_s1_ids = set(s1_df["entity_id"].astype(str).str.strip()) if sample_size is not None else None
        sub_validator = SubmissionValidator()
        sub_result = sub_validator.validate_submission(
            matching_path=matching_output_path,
            candidate_path=candidate_output_path,
            test_dir=test_dir,
            expected_s1_ids=required_s1_ids,
            check_target_ids=False,
        )

        print("\n" + sub_result.report() + "\n")

        if not sub_result.is_valid:
            raise ValueError(f"Submission validation failed: {sub_result.errors}")

        return sub_result


def run_cli_inference():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution inference on test data")
    parser.add_argument("--test-dir", type=str, default="data/test", help="Test dataset directory")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Train dataset directory")
    parser.add_argument("--model-path", type=str, default="models/best_model.joblib", help="Model artifact path")
    parser.add_argument("--threshold-path", type=str, default="models/best_threshold.json", help="Threshold JSON path")
    parser.add_argument("--matching-output", type=str, default="outputs/matching_results.tsv", help="Matching TSV output")
    parser.add_argument("--candidate-output", type=str, default="outputs/candidate_pairs.tsv", help="Candidate TSV output")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample limit for Source 1 test entities")
    parser.add_argument("--target-sample-size", type=int, default=None, help="Optional target sample limit")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for streaming processing")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit decision threshold override")
    args = parser.parse_args()

    pipeline = InferencePipeline(
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        threshold=args.threshold,
    )
    pipeline.run_full_inference(
        test_dir=args.test_dir,
        train_dir=args.train_dir,
        matching_output_path=args.matching_output,
        candidate_output_path=args.candidate_output,
        sample_size=args.sample_size,
        target_sample_size=args.target_sample_size,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    run_cli_inference()
