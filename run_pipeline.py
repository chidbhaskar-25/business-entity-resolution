"""End-to-End Execution Pipeline for Business Entity Resolution.

Amazon ML Challenge 2026

Usage:
    # Run full end-to-end pipeline (data check, train, threshold search, inference, validation)
    python run_pipeline.py --mode all

    # Run full pipeline with custom test sample size
    python run_pipeline.py --mode all --sample-size 1000

    # Run on full test dataset without sampling
    python run_pipeline.py --mode all --full

    # Specific pipeline stages
    python run_pipeline.py --mode train
    python run_pipeline.py --mode evaluate
    python run_pipeline.py --mode predict
    python run_pipeline.py --mode validate
    python run_pipeline.py --mode experiments
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# Ensure standard UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

from src.business_entity_resolution.data.loader import DataLoader
from src.business_entity_resolution.data.submission_validator import SubmissionValidator
from src.business_entity_resolution.evaluation.threshold_optimizer import (
    ThresholdOptimizer,
    run_threshold_optimization_pipeline,
)
from src.business_entity_resolution.inference.pipeline import InferencePipeline
from src.business_entity_resolution.models.train_model import train_baseline
from src.business_entity_resolution.utils.logger import setup_logger

logger = setup_logger("run_pipeline")


def validate_environment(
    train_dir: str = "data/train",
    test_dir: str = "data/test",
) -> None:
    """Verifies existence of required data directories and files."""
    logger.info("----------------------------------------------------------------")
    logger.info("STAGE 1: VERIFYING DATASETS & ENVIRONMENT")
    logger.info("----------------------------------------------------------------")

    required_train_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ]
    required_test_files = [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]

    for fname in required_train_files:
        fpath = os.path.join(train_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing required training file: {fpath}")
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.info(f"  [OK] Train source: {fname} ({size_mb:.1f} MB)")

    for fname in required_test_files:
        fpath = os.path.join(test_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing required test file: {fpath}")
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.info(f"  [OK] Test source:  {fname} ({size_mb:.1f} MB)")


def prepare_training_dataset(
    dataset_path: str = "outputs/train_dataset.npz",
    rebuild: bool = False,
    s1_sample_size: int = 5000,
    target_sample_size: int = 100000,
) -> str:
    """Ensures training dataset NPZ is ready, building if missing or requested."""
    logger.info("----------------------------------------------------------------")
    logger.info("STAGE 2: PREPARING PAIRWISE TRAINING DATASET")
    logger.info("----------------------------------------------------------------")

    if os.path.isfile(dataset_path) and not rebuild:
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        logger.info(f"Using existing training dataset: {dataset_path} ({size_mb:.2f} MB)")
        return dataset_path

    logger.info("Building new training dataset from training ground truth...")
    from build_train_dataset import build_and_save_training_dataset

    return build_and_save_training_dataset(
        train_dir="data/train",
        s1_sample_size=s1_sample_size,
        target_sample_size=target_sample_size,
        max_candidates_per_entity=25,
        output_npz_path=dataset_path,
    )


def train_matching_model(
    dataset_path: str = "outputs/train_dataset.npz",
    model_path: str = "models/best_model.joblib",
    metrics_path: str = "outputs/training_metrics.json",
    model_type: str = "hist_gradient_boosting",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Trains matching model and saves artifact."""
    logger.info("----------------------------------------------------------------")
    logger.info("STAGE 3: TRAINING ENTITY MATCHING MODEL")
    logger.info("----------------------------------------------------------------")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    metrics = train_baseline(
        dataset_path=dataset_path,
        output_model_path=model_path,
        metrics_output_path=metrics_path,
        model_type=model_type,
        val_size=0.20,
        random_seed=random_seed,
        class_weight="balanced",
    )
    return metrics


def optimize_threshold(
    dataset_path: str = "outputs/train_dataset.npz",
    model_path: str = "models/best_model.joblib",
    threshold_json_path: str = "models/best_threshold.json",
    experiment_report_path: str = "outputs/threshold_experiment.json",
    config_path: str = "config.yaml",
) -> float:
    """Sweeps decision thresholds to maximize Macro F0.5."""
    logger.info("----------------------------------------------------------------")
    logger.info("STAGE 4: DECISION THRESHOLD OPTIMIZATION (MACRO F0.5)")
    logger.info("----------------------------------------------------------------")

    report = run_threshold_optimization_pipeline(
        dataset_path=dataset_path,
        model_path=model_path,
        config_path=config_path,
        threshold_json_path=threshold_json_path,
        experiment_report_path=experiment_report_path,
        min_threshold=0.10,
        max_threshold=0.95,
        step=0.01,
        use_validation_split=True,
    )
    # Also save to models/threshold.json for backwards compatibility
    ThresholdOptimizer.save_threshold_json(report, filepath="models/threshold.json")
    ThresholdOptimizer.update_config_yaml(report.optimal_threshold, config_path=config_path)

    logger.info(
        f"Optimal threshold found: tau = {report.optimal_threshold:.2f} "
        f"(Macro F0.5: {report.best_macro_f05:.4f}, Precision: {report.best_macro_precision:.4f}, Recall: {report.best_macro_recall:.4f})"
    )
    return report.optimal_threshold


def run_inference(
    test_dir: str = "data/test",
    train_dir: str = "data/train",
    model_path: str = "models/best_model.joblib",
    threshold_path: str = "models/best_threshold.json",
    matching_output: str = "outputs/matching_results.tsv",
    candidate_output: str = "outputs/candidate_pairs.tsv",
    sample_size: Optional[int] = 500,
    target_sample_size: Optional[int] = 50000,
    chunk_size: int = 5000,
    threshold: Optional[float] = None,
) -> Any:
    """Executes inference engine generating competition TSV outputs."""
    logger.info("----------------------------------------------------------------")
    logger.info("STAGE 5: INFERENCE & COMPETITION OUTPUT GENERATION")
    logger.info("----------------------------------------------------------------")

    pipeline = InferencePipeline(
        model_path=model_path,
        threshold_path=threshold_path,
        threshold=threshold,
    )

    report = pipeline.run_full_inference(
        test_dir=test_dir,
        train_dir=train_dir,
        matching_output_path=matching_output,
        candidate_output_path=candidate_output,
        sample_size=sample_size,
        target_sample_size=target_sample_size,
        chunk_size=chunk_size,
    )

    # Mirror files to ensure both output/ and outputs/ are populated
    import shutil
    os.makedirs("output", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    for src, dst in [
        (matching_output, os.path.join("output", "matching_results.tsv")),
        (matching_output, os.path.join("outputs", "matching_results.tsv")),
        (candidate_output, os.path.join("output", "candidate_pairs.tsv")),
        (candidate_output, os.path.join("outputs", "candidate_pairs.tsv")),
    ]:
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

    return report


def validate_submission_outputs(
    matching_path: str = "outputs/matching_results.tsv",
    candidate_path: str = "outputs/candidate_pairs.tsv",
    test_dir: str = "data/test",
    sample_size: Optional[int] = None,
    check_target_ids: bool = False,
) -> bool:
    """Strictly validates generated TSV files across all 11 criteria."""
    logger.info("----------------------------------------------------------------")
    logger.info("STAGE 6: STRICT SUBMISSION FORMAT VALIDATION")
    logger.info("----------------------------------------------------------------")

    expected_s1 = None
    if sample_size is not None:
        loader = DataLoader(test_dir)
        s1_df = loader.load_source_df("test_source1.tsv", nrows=sample_size)
        expected_s1 = set(s1_df["entity_id"].astype(str).str.strip())

    validator = SubmissionValidator()
    result = validator.validate_submission(
        matching_path=matching_path,
        candidate_path=candidate_path,
        test_dir=test_dir,
        expected_s1_ids=expected_s1,
        check_target_ids=check_target_ids,
    )

    print("\n" + result.report() + "\n")
    return result.is_valid


def main():
    parser = argparse.ArgumentParser(
        description="Run Business Entity Resolution pipeline stages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train", "evaluate", "predict", "validate", "experiments"],
        help="Pipeline mode to execute",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Sample size for test Source 1 entities during inference (use --full for all)",
    )
    parser.add_argument(
        "--target-sample-size",
        type=int,
        default=50000,
        help="Target records sample limit for inference (use --full for all targets)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run inference over entire un-sampled test set (1.73M entities)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hist_gradient_boosting",
        choices=["hist_gradient_boosting", "logistic_regression"],
        help="Matching model architecture",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.joblib",
        help="Path to serialized model artifact",
    )
    parser.add_argument(
        "--threshold-path",
        type=str,
        default="models/best_threshold.json",
        help="Path to threshold JSON configuration",
    )
    parser.add_argument(
        "--matching-output",
        type=str,
        default="outputs/matching_results.tsv",
        help="Output destination for matching_results.tsv",
    )
    parser.add_argument(
        "--candidate-output",
        type=str,
        default="outputs/candidate_pairs.tsv",
        help="Output destination for candidate_pairs.tsv",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Reconstruct pairwise training dataset even if already exists",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit decision threshold override",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Batch chunk size for streaming inference",
    )

    args = parser.parse_args()

    # Determine effective inference sample sizes
    effective_s1_sample = None if args.full else args.sample_size
    effective_target_sample = None if args.full else args.target_sample_size

    print("\n" + "=" * 80)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 80)
    print(f"Mode:               {args.mode.upper()}")
    print(f"Model Type:         {args.model_type}")
    print(f"Model Path:         {args.model_path}")
    print(f"Threshold Path:     {args.threshold_path}")
    print(f"Matching Output:    {args.matching_output}")
    print(f"Candidate Output:   {args.candidate_output}")
    print(f"Test S1 Sample:     {'FULL (1.73M entities)' if effective_s1_sample is None else f'{effective_s1_sample:,} entities'}")
    print(f"Test Target Sample: {'FULL (~10M entities)' if effective_target_sample is None else f'{effective_target_sample:,} entities'}")
    print("=" * 80 + "\n")

    t_start = time.time()

    # Execute requested mode
    if args.mode == "validate":
        validate_submission_outputs(
            matching_path=args.matching_output,
            candidate_path=args.candidate_output,
            sample_size=effective_s1_sample,
        )
    elif args.mode == "experiments":
        from run_experiments import run_controlled_experiments

        run_controlled_experiments()
    elif args.mode == "train":
        validate_environment()
        dataset_path = prepare_training_dataset(rebuild=args.rebuild_dataset)
        train_matching_model(
            dataset_path=dataset_path,
            model_path=args.model_path,
            model_type=args.model_type,
        )
    elif args.mode == "evaluate":
        validate_environment()
        dataset_path = prepare_training_dataset(rebuild=args.rebuild_dataset)
        optimize_threshold(
            dataset_path=dataset_path,
            model_path=args.model_path,
            threshold_json_path=args.threshold_path,
        )
    elif args.mode == "predict":
        validate_environment()
        run_inference(
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            matching_output=args.matching_output,
            candidate_output=args.candidate_output,
            sample_size=effective_s1_sample,
            target_sample_size=effective_target_sample,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
        )
    elif args.mode == "all":
        # Full end-to-end pipeline
        validate_environment()
        dataset_path = prepare_training_dataset(rebuild=args.rebuild_dataset)
        train_matching_model(
            dataset_path=dataset_path,
            model_path=args.model_path,
            model_type=args.model_type,
        )
        opt_th = optimize_threshold(
            dataset_path=dataset_path,
            model_path=args.model_path,
            threshold_json_path=args.threshold_path,
        )
        run_inference(
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            matching_output=args.matching_output,
            candidate_output=args.candidate_output,
            sample_size=effective_s1_sample,
            target_sample_size=effective_target_sample,
            chunk_size=args.chunk_size,
            threshold=args.threshold or opt_th,
        )
        validate_submission_outputs(
            matching_path=args.matching_output,
            candidate_path=args.candidate_output,
            sample_size=effective_s1_sample,
        )

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Total Elapsed Time: {elapsed:.2f} seconds")
    print("=" * 80)

    # Confirm generated output files
    if os.path.exists(args.matching_output) and os.path.exists(args.candidate_output):
        m_size = os.path.getsize(args.matching_output)
        c_size = os.path.getsize(args.candidate_output)
        with open(args.matching_output, "r", encoding="utf-8") as f:
            m_rows = sum(1 for _ in f) - 1
        with open(args.candidate_output, "r", encoding="utf-8") as f:
            c_rows = sum(1 for _ in f) - 1
        print(f"\nFinal Generated Outputs:")
        print(f"  [OK] {args.matching_output} ({m_size:,} bytes, {m_rows:,} entities)")
        print(f"  [OK] {args.candidate_output} ({c_size:,} bytes, {c_rows:,} entities)")
    print("\n")


if __name__ == "__main__":
    main()
