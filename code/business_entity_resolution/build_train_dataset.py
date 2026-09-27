"""Constructs and validates the pairwise training dataset from training ground truth."""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding="utf-8")

from src.business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from src.business_entity_resolution.data.label_builder import LabelBuilder
from src.business_entity_resolution.data.loader import DataLoader
from src.business_entity_resolution.features.feature_builder import FeatureBuilder
from src.business_entity_resolution.utils.logger import setup_logger

logger = setup_logger("build_train_dataset")


def build_and_save_training_dataset(
    train_dir: str = "data/train",
    s1_sample_size: int = 5000,
    target_sample_size: int = 100000,
    max_candidates_per_entity: int = 25,
    output_npz_path: str = "outputs/train_dataset.npz",
):
    logger.info("================================================================")
    logger.info("BUILDING GROUND-TRUTH TRAINING DATASET")
    logger.info("================================================================")

    loader = DataLoader(train_dir)

    # 1. Load Ground Truth
    logger.info(f"Loading ground truth labels (limit={s1_sample_size})...")
    gt_map = loader.load_ground_truth("train_ground_truth.tsv", nrows=s1_sample_size, validate=True)
    sampled_s1_ids = set(gt_map.keys())

    # Count singletons in sample
    singleton_count = sum(1 for mids in gt_map.values() if not mids)
    logger.info(
        f"Sample contains {len(gt_map):,} S1 entities ({singleton_count:,} singletons / {singleton_count / len(gt_map):.2%})"
    )

    # 2. Load Source 1 records
    logger.info("Loading corresponding Source 1 records...")
    s1_records_list = []
    for chunk in loader.stream_source_chunks("train_source1.tsv", chunk_size=50000):
        matched_chunk = chunk[chunk["entity_id"].isin(sampled_s1_ids)]
        if not matched_chunk.empty:
            s1_records_list.append(matched_chunk)
        if sum(len(c) for c in s1_records_list) >= len(sampled_s1_ids):
            break

    s1_df = pd.concat(s1_records_list, ignore_index=True) if s1_records_list else pd.DataFrame()
    logger.info(f"Loaded {len(s1_df):,} Source 1 entities.")

    # 3. Load Target records (S2 + S3)
    needed_target_ids = set()
    for s1_id in s1_df["entity_id"]:
        needed_target_ids.update(gt_map.get(s1_id, set()))

    logger.info(f"Collecting target records (including {len(needed_target_ids):,} true targets)...")
    s2_df = loader.load_source_df("train_source2.tsv", nrows=target_sample_size)
    s3_df = loader.load_source_df("train_source3.tsv", nrows=target_sample_size)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    found_targets = set(target_df["entity_id"])
    missing_targets = needed_target_ids - found_targets
    if missing_targets:
        logger.info(f"Streaming {len(missing_targets):,} required ground truth targets into target pool...")
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

    logger.info(f"Target pool contains {len(target_df):,} records.")

    # 4. Generate Candidates via Blocking
    generator = CandidateGenerator(max_candidates_per_entity=max_candidates_per_entity)
    generator.fit_target_records(target_df)
    candidate_map, blocking_metrics = generator.generate_candidates_batch(s1_df, ground_truth=gt_map)

    # 5. Build S1 and Target Lookups
    s1_lookup = {r["entity_id"]: r for r in s1_df.to_dict(orient="records")}
    target_lookup = {r["entity_id"]: r for r in target_df.to_dict(orient="records")}

    # 6. Fit Feature Builder TF-IDF strictly on training reference data (NO LEAKAGE)
    feature_builder = FeatureBuilder()
    feature_builder.fit_reference_tfidf(
        s1_names=s1_df["business_name"].tolist(),
        s1_addresses=s1_df["business_address"].tolist(),
    )

    # 7. Build Ground-Truth Labels & Feature Matrix
    t0 = time.time()
    X, y, valid_pairs, validation_report = LabelBuilder.build_training_dataset(
        candidate_map=candidate_map,
        ground_truth_map=gt_map,
        s1_lookup=s1_lookup,
        target_lookup=target_lookup,
        feature_builder=feature_builder,
    )
    build_time = time.time() - t0

    # 8. Save Compressed Dataset
    LabelBuilder.save_dataset_npz(
        filepath=output_npz_path,
        X=X,
        y=y,
        pairs=valid_pairs,
        feature_names=feature_builder.feature_names,
    )

    # 9. Verify reload
    X_test, y_test, pairs_test, names_test = LabelBuilder.load_dataset_npz(output_npz_path)
    assert len(X_test) == len(X)
    assert len(y_test) == len(y)
    assert len(pairs_test) == len(valid_pairs)

    print("\n" + "=" * 80)
    print("TRAINING DATASET GENERATION & VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Candidate Pairs:      {validation_report.total_pairs:,}")
    print(f"Positive Matches (y = 1):   {validation_report.positive_labels:,} ({validation_report.positive_ratio:.2%})")
    print(f"Negative Matches (y = 0):   {validation_report.negative_labels:,} ({1 - validation_report.positive_ratio:.2%})")
    print(f"Singleton Pairs:            {validation_report.singleton_pairs_count:,}")
    print(f"Singleton Positive Errors:  {validation_report.singleton_positive_count} (Must be strictly 0)")
    print(f"Feature Matrix Shape:       {X.shape} (float32)")
    print(f"Label Vector Shape:         {y.shape} (int32)")
    print(f"Compressed File Size:       {os.path.getsize(output_npz_path) / (1024 * 1024):.2f} MB")
    print(f"Output File:                {output_npz_path}")
    print(f"Validation Status:          {'PASSED (100% Valid)' if validation_report.is_valid else 'FAILED'}")
    print("=" * 80 + "\n")

    return output_npz_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-sample", type=int, default=5000, help="S1 sample size")
    parser.add_argument("--target-sample", type=int, default=100000, help="Target pool sample size")
    parser.add_argument("--max-candidates", type=int, default=25, help="Max candidates per S1")
    parser.add_argument("--output", type=str, default="outputs/train_dataset.npz", help="Output path")
    args = parser.parse_args()

    build_and_save_training_dataset(
        train_dir="data/train",
        s1_sample_size=args.s1_sample,
        target_sample_size=args.target_sample,
        max_candidates_per_entity=args.max_candidates,
        output_npz_path=args.output,
    )
