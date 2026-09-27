"""Script to evaluate candidate generation and blocking recall on training data."""

import argparse
import os
import sys
import time
import pandas as pd

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding="utf-8")

from src.business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from src.business_entity_resolution.data.loader import DataLoader
from src.business_entity_resolution.utils.logger import setup_logger

logger = setup_logger("evaluate_blocking")


def evaluate_blocking_on_training_data(
    train_dir: str = "data/train",
    s1_sample_size: int = 5000,
    target_sample_size: int = 50000,
    max_candidates_per_entity: int = 25,
    output_path: str = "outputs/blocking_metrics.json",
):
    logger.info("================================================================")
    logger.info("EVALUATING CANDIDATE GENERATION & BLOCKING RECALL")
    logger.info("================================================================")

    loader = DataLoader(train_dir)

    # 1. Load Ground Truth mapping
    logger.info(f"Loading ground truth labels (limit={s1_sample_size})...")
    gt_map = loader.load_ground_truth("train_ground_truth.tsv", nrows=s1_sample_size)
    sampled_s1_ids = set(gt_map.keys())

    # 2. Load matching S1 records
    logger.info("Loading corresponding Source 1 records...")
    s1_records_list = []
    for chunk in loader.stream_source_chunks("train_source1.tsv", chunk_size=50000):
        matched_chunk = chunk[chunk["entity_id"].isin(sampled_s1_ids)]
        if not matched_chunk.empty:
            s1_records_list.append(matched_chunk)
        if sum(len(c) for c in s1_records_list) >= len(sampled_s1_ids):
            break

    s1_df = pd.concat(s1_records_list, ignore_index=True) if s1_records_list else pd.DataFrame()
    logger.info(f"Loaded {len(s1_df):,} Source 1 entities for evaluation.")

    # 3. Load Target records (S2 and S3)
    logger.info(f"Loading target records from S2 and S3 (target_sample_size={target_sample_size})...")
    # Also collect any ground-truth target IDs that belong to the sampled S1 to test recall realistically
    needed_target_ids = set()
    for s1_id in s1_df["entity_id"]:
        needed_target_ids.update(gt_map.get(s1_id, set()))

    logger.info(f"Total true ground-truth targets to capture: {len(needed_target_ids):,}")

    # Load targets: sample plus specific ground truth targets to ensure realistic target population
    s2_df = loader.load_source_df("train_source2.tsv", nrows=target_sample_size)
    s3_df = loader.load_source_df("train_source3.tsv", nrows=target_sample_size)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Check how many needed targets are in target_df, and if missing, load their chunks
    found_targets = set(target_df["entity_id"])
    missing_targets = needed_target_ids - found_targets
    if missing_targets:
        logger.info(f"Retrieving {len(missing_targets):,} true target records from sources...")
        extra_targets = []
        for src_file in ["train_source2.tsv", "train_source3.tsv"]:
            for chunk in loader.stream_source_chunks(src_file, chunk_size=100000):
                sub = chunk[chunk["entity_id"].isin(missing_targets)]
                if not sub.empty:
                    extra_targets.append(sub)
                    missing_targets -= set(sub["entity_id"])
                if not missing_targets:
                    break
        if extra_targets:
            target_df = pd.concat([target_df] + extra_targets, ignore_index=True)

    logger.info(f"Target pool contains {len(target_df):,} records (S2 + S3).")

    # 4. Fit Candidate Generator
    t0 = time.time()
    generator = CandidateGenerator(max_candidates_per_entity=max_candidates_per_entity)
    generator.fit_target_records(target_df)

    # 5. Run Candidate Generation & Measure Metrics
    logger.info("Generating candidate pairs across multi-strategy blockers...")
    cands_map, metrics = generator.generate_candidates_batch(s1_df, ground_truth=gt_map)
    duration = time.time() - t0

    # 6. Save Metrics
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator.save_metrics(metrics, output_path)

    # 7. Print Comprehensive Summary
    print("\n" + "=" * 80)
    print("CANDIDATE GENERATION & BLOCKING EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Source 1 Records:          {metrics.total_source1_records:,}")
    print(f"Total Target Records (S2 + S3):  {metrics.total_target_records:,}")
    print(f"Total Candidates Generated:      {metrics.total_candidate_pairs:,}")
    print(f"Average Candidates per Source 1: {metrics.avg_candidates_per_source1:.2f}")
    print(f"True Pairs Found in Candidates:  {metrics.true_pairs_found:,} / {metrics.total_true_pairs:,}")
    print(f"Candidate Recall:                {metrics.candidate_recall:.2%}")
    print(f"Reduction Ratio:                 {metrics.reduction_ratio:.6f} ({metrics.reduction_ratio * 100:.4f}%)")
    print(f"Elapsed Evaluation Time:         {duration:.2f} seconds")
    print(f"Saved to:                        {output_path}")
    print("=" * 80 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-sample", type=int, default=5000, help="Number of S1 records to evaluate")
    parser.add_argument("--target-sample", type=int, default=100000, help="Target records sample size")
    parser.add_argument("--max-candidates", type=int, default=25, help="Max candidates per S1 entity")
    parser.add_argument("--output", type=str, default="outputs/blocking_metrics.json", help="Output JSON path")
    args = parser.parse_args()

    evaluate_blocking_on_training_data(
        s1_sample_size=args.s1_sample,
        target_sample_size=args.target_sample,
        max_candidates_per_entity=args.max_candidates,
        output_path=args.output,
    )
