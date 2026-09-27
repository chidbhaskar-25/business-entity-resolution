"""Streaming and batch output writer for competition TSV submission files."""

from collections import OrderedDict
import os
from typing import Dict, Iterable, List, Optional, Set, TextIO, Tuple
from ..utils.logger import setup_logger

logger = setup_logger("inference.output_writer")

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


class SubmissionWriter:
    """Manages creation, formatting, and strict validation of submission TSV files.

    Enforces all challenge constraints:
    - Tab-separated format (.tsv) with exact required column headers.
    - Exactly one row per Source 1 entity.
    - Deduplicated target IDs preserving order.
    - Strict S2-/S3- prefix filtering (zero S1- self-matches).
    - Match list is guaranteed to be a subset of candidate pairs.
    - Singletons are correctly formatted as empty match lists (`s1_id\\t\\n`).
    """

    def __init__(
        self,
        matching_path: str,
        candidate_path: str,
        enforce_subset: bool = True,
    ):
        self.matching_path = matching_path
        self.candidate_path = candidate_path
        self.enforce_subset = enforce_subset

        self._f_match: Optional[TextIO] = None
        self._f_cand: Optional[TextIO] = None
        self._written_s1_ids: Set[str] = set()

        self.stats = {
            "total_s1_written": 0,
            "total_matches": 0,
            "total_candidates": 0,
            "singleton_count": 0,
            "multi_match_count": 0,
            "filtered_self_matches": 0,
            "filtered_bad_prefixes": 0,
        }

    def __enter__(self) -> "SubmissionWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        """Opens output files and writes exact TSV headers."""
        os.makedirs(os.path.dirname(os.path.abspath(self.matching_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.candidate_path)), exist_ok=True)

        self._f_match = open(self.matching_path, "w", encoding="utf-8", newline="\n")
        self._f_cand = open(self.candidate_path, "w", encoding="utf-8", newline="\n")

        # Write exact required headers
        self._f_match.write(MATCHING_HEADER)
        self._f_cand.write(CANDIDATE_HEADER)
        self._written_s1_ids.clear()
        logger.info(f"Initialized submission writer for {self.matching_path} and {self.candidate_path}")

    def close(self) -> None:
        """Flushes and closes file handles."""
        if self._f_match and not self._f_match.closed:
            self._f_match.close()
        if self._f_cand and not self._f_cand.closed:
            self._f_cand.close()

    def _clean_target_ids(
        self,
        raw_ids: Iterable[str],
        s1_id: str,
    ) -> List[str]:
        """Validates, filters, and deduplicates target IDs in rank order."""
        seen = set()
        clean = []
        for tid in raw_ids:
            tid_str = str(tid).strip()
            if not tid_str:
                continue
            if tid_str.startswith("S1-"):
                self.stats["filtered_self_matches"] += 1
                logger.warning(f"Filtered invalid self-match '{tid_str}' for Source 1 entity '{s1_id}'.")
                continue
            if not (tid_str.startswith("S2-") or tid_str.startswith("S3-")):
                self.stats["filtered_bad_prefixes"] += 1
                logger.warning(f"Filtered invalid prefix in target ID '{tid_str}' for '{s1_id}'.")
                continue
            if tid_str not in seen:
                seen.add(tid_str)
                clean.append(tid_str)
        return clean

    def write_entity_result(
        self,
        s1_id: str,
        matched_ids: Iterable[str],
        candidate_ids: Iterable[str],
    ) -> None:
        """Writes matching and candidate rows for a single Source 1 entity.

        Args:
            s1_id: Source 1 entity ID.
            matched_ids: Final predicted matches for this entity.
            candidate_ids: Blocking candidate pairs for this entity.

        Raises:
            ValueError: If s1_id is duplicate or invalid.
        """
        sid = str(s1_id).strip()
        if not sid:
            raise ValueError("Empty Source 1 entity ID provided.")
        if sid in self._written_s1_ids:
            raise ValueError(f"Duplicate Source 1 entity ID detected: '{sid}'. Exactly one row allowed per entity.")

        # 1. Clean candidates first
        clean_cands = self._clean_target_ids(candidate_ids, sid)

        # 2. Clean matches
        clean_matches = self._clean_target_ids(matched_ids, sid)

        # 3. Enforce matches are subset of candidates
        cand_set = set(clean_cands)
        if self.enforce_subset:
            # If any predicted match was somehow omitted from candidates, include it in candidates
            for mid in clean_matches:
                if mid not in cand_set:
                    clean_cands.append(mid)
                    cand_set.add(mid)

        # Format rows: tab-delimited, comma-separated IDs without spaces
        match_str = ",".join(clean_matches)
        cand_str = ",".join(clean_cands)

        self._f_match.write(f"{sid}\t{match_str}\n")
        self._f_cand.write(f"{sid}\t{cand_str}\n")

        self._written_s1_ids.add(sid)
        self.stats["total_s1_written"] += 1
        self.stats["total_matches"] += len(clean_matches)
        self.stats["total_candidates"] += len(clean_cands)

        if len(clean_matches) == 0:
            self.stats["singleton_count"] += 1
        elif len(clean_matches) > 1:
            self.stats["multi_match_count"] += 1

    def write_batch(
        self,
        s1_ids: List[str],
        matches_by_s1: Dict[str, List[str]],
        candidates_by_s1: Dict[str, List[str]],
    ) -> None:
        """Writes results for a batch of Source 1 entities."""
        for sid in s1_ids:
            m_list = matches_by_s1.get(sid, [])
            c_list = candidates_by_s1.get(sid, [])
            self.write_entity_result(sid, m_list, c_list)

    @classmethod
    def write_submission_files(
        cls,
        s1_ids: List[str],
        matches_by_s1: Dict[str, List[str]],
        candidates_by_s1: Dict[str, List[str]],
        matching_path: str = "outputs/matching_results.tsv",
        candidate_path: str = "outputs/candidate_pairs.tsv",
    ) -> Dict[str, int]:
        """Convenience method to write full submission files from in-memory mappings."""
        writer = cls(matching_path=matching_path, candidate_path=candidate_path)
        with writer:
            writer.write_batch(
                s1_ids=s1_ids,
                matches_by_s1=matches_by_s1,
                candidates_by_s1=candidates_by_s1,
            )
        return writer.stats
