"""Multi-strategy candidate generator and blocking evaluation."""

from collections import Counter, defaultdict
import json
import os
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from pydantic import BaseModel

from .blocker import CompositeBlocker
from ..utils.logger import setup_logger

logger = setup_logger("blocking.candidate_generator")


class BlockingMetrics(BaseModel):
    """Container for candidate generation evaluation metrics."""
    total_source1_records: int
    total_target_records: int
    total_candidate_pairs: int
    avg_candidates_per_source1: float
    candidate_recall: Optional[float] = None
    reduction_ratio: float
    true_pairs_found: Optional[int] = None
    total_true_pairs: Optional[int] = None


class CandidateGenerator:
    """Manages inverted index creation, candidate pair retrieval, and metric tracking."""

    def __init__(
        self,
        blocker: Optional[CompositeBlocker] = None,
        max_candidates_per_entity: int = 25,
    ):
        self.blocker = blocker or CompositeBlocker()
        self.max_candidates_per_entity = max_candidates_per_entity

    def fit_target_records(self, target_df: pd.DataFrame) -> "CandidateGenerator":
        """Indexes target records (Source 2 and Source 3) into country partitions.

        Args:
            target_df: DataFrame containing entity_id, business_name, business_address, country.

        Returns:
            self
        """
        logger.info(f"Indexing {len(target_df):,} target records across country partitions...")
        self.blocker.inverted_index.clear()
        self.blocker.total_target_records = 0

        for _, row in target_df.iterrows():
            eid = str(row["entity_id"]).strip()
            name = str(row.get("business_name", ""))
            addr = str(row.get("business_address", ""))
            country = str(row.get("country", ""))
            self.blocker.index_target_record(eid, name, addr, country)

        num_keys = sum(len(idx) for idx in self.blocker.inverted_index.values())
        logger.info(
            f"Successfully indexed {self.blocker.total_target_records:,} target records across "
            f"{len(self.blocker.inverted_index)} country partitions. Total unique blocking keys: {num_keys:,}"
        )
        return self

    def generate_candidates_for_entity(
        self,
        name: str,
        address: str,
        country: str,
    ) -> List[str]:
        """Generates ranked candidate target IDs for a single Source 1 entity.

        Ranks candidates by number of matched blocking keys, weighting exact name and
        address number matches higher.
        """
        c_label = self.blocker.normalize_country_label(country)
        country_index = self.blocker.inverted_index.get(c_label)
        if not country_index:
            return []

        keys = self.blocker.generate_record_keys(name, address)
        if not keys:
            return []

        scores = Counter()
        for k in keys:
            if k not in country_index:
                continue
            # Exact name or bigram receives higher priority weight
            if k.startswith("name_exact:"):
                weight = 4
            elif k.startswith("name_bi:"):
                weight = 3
            elif k.startswith("addr_num:"):
                weight = 2
            else:
                weight = 1

            for tid in country_index[k]:
                scores[tid] += weight

        if not scores:
            return []

        # Return top N candidates ordered by priority score
        ranked_candidates = [
            tid for tid, _ in scores.most_common(self.max_candidates_per_entity)
        ]
        return ranked_candidates

    def generate_candidates_batch(
        self,
        s1_df: pd.DataFrame,
        ground_truth: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[Dict[str, List[str]], BlockingMetrics]:
        """Generates candidates for a collection of Source 1 entities and calculates metrics.

        Args:
            s1_df: DataFrame of Source 1 entities.
            ground_truth: Optional mapping {s1_id: set(target_ids)} to evaluate candidate recall.

        Returns:
            Tuple of (candidate_map: {s1_id: [cand_ids]}, metrics: BlockingMetrics).
        """
        logger.info(f"Generating candidates for {len(s1_df):,} Source 1 entities...")
        candidate_map: Dict[str, List[str]] = {}

        total_candidates = 0
        true_pairs_found = 0
        total_true_pairs = 0

        for _, row in s1_df.iterrows():
            sid = str(row["entity_id"]).strip()
            name = str(row.get("business_name", ""))
            addr = str(row.get("business_address", ""))
            country = str(row.get("country", ""))

            cands = self.generate_candidates_for_entity(name, addr, country)
            candidate_map[sid] = cands
            total_candidates += len(cands)

            if ground_truth and sid in ground_truth:
                true_targets = ground_truth[sid]
                if true_targets:
                    total_true_pairs += len(true_targets)
                    true_pairs_found += len(set(cands) & true_targets)

        num_s1 = len(s1_df)
        num_targets = max(self.blocker.total_target_records, 1)
        avg_cands = total_candidates / max(num_s1, 1)

        total_possible = num_s1 * num_targets
        reduction_ratio = 1.0 - (total_candidates / max(total_possible, 1))

        cand_recall = None
        if total_true_pairs > 0:
            cand_recall = true_pairs_found / total_true_pairs

        metrics = BlockingMetrics(
            total_source1_records=num_s1,
            total_target_records=self.blocker.total_target_records,
            total_candidate_pairs=total_candidates,
            avg_candidates_per_source1=round(avg_cands, 2),
            candidate_recall=round(cand_recall, 4) if cand_recall is not None else None,
            reduction_ratio=round(reduction_ratio, 6),
            true_pairs_found=true_pairs_found if total_true_pairs > 0 else None,
            total_true_pairs=total_true_pairs if total_true_pairs > 0 else None,
        )

        logger.info(
            f"Blocking results: {total_candidates:,} total candidate pairs. "
            f"Avg candidates/S1: {avg_cands:.2f}. Reduction ratio: {reduction_ratio:.6f}. "
            + (f"Candidate Recall: {cand_recall:.2%} ({true_pairs_found:,}/{total_true_pairs:,} true links)" if cand_recall is not None else "")
        )
        return candidate_map, metrics

    def save_metrics(self, metrics: BlockingMetrics, filepath: str = "outputs/blocking_metrics.json") -> None:
        """Saves blocking metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics.model_dump(), f, indent=2)
        logger.info(f"Saved blocking metrics to: {filepath}")
