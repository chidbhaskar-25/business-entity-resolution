"""Comprehensive unit tests for SubmissionValidator and SubmissionWriter."""

import os
import pytest

from src.business_entity_resolution.data.submission_validator import (
    SubmissionValidationResult,
    SubmissionValidator,
)
from src.business_entity_resolution.inference.output_writer import SubmissionWriter


def test_submission_writer_and_validator_pass_flow(tmp_path):
    matching_tsv = str(tmp_path / "matching_results.tsv")
    candidate_tsv = str(tmp_path / "candidate_pairs.tsv")

    s1_ids = ["S1-100", "S1-101", "S1-102", "S1-103"]
    matches = {
        "S1-100": ["S2-200", "S3-300"],
        "S1-101": ["S2-201"],
        "S1-102": [],  # Singleton
        "S1-103": ["S3-303"],
    }
    candidates = {
        "S1-100": ["S2-200", "S3-300", "S2-205"],
        "S1-101": ["S2-201", "S3-305"],
        "S1-102": ["S2-209"],  # Candidates exist, but no matches (singleton)
        "S1-103": ["S3-303"],
    }

    # Write files using SubmissionWriter
    stats = SubmissionWriter.write_submission_files(
        s1_ids=s1_ids,
        matches_by_s1=matches,
        candidates_by_s1=candidates,
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
    )

    assert stats["total_s1_written"] == 4
    assert stats["singleton_count"] == 1
    assert stats["multi_match_count"] == 1

    # Validate
    validator = SubmissionValidator()
    result = validator.validate_submission(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        expected_s1_ids=set(s1_ids),
    )

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.criteria_checks["correct_tsv_format"] is True
    assert result.criteria_checks["exact_column_names"] is True
    assert result.criteria_checks["every_s1_test_id_exists"] is True
    assert result.criteria_checks["exactly_one_row_per_s1"] is True
    assert result.criteria_checks["no_duplicate_s1_rows"] is True
    assert result.criteria_checks["only_s2_s3_matches"] is True
    assert result.criteria_checks["no_duplicate_ids_in_lists"] is True
    assert result.criteria_checks["candidate_ids_valid"] is True
    assert result.criteria_checks["matches_subset_of_candidates"] is True
    assert result.criteria_checks["empty_lists_correctly_represented"] is True

    # Check report string
    report_text = result.report()
    assert "PASS (100% Compliant - Safe for Leaderboard Submission)" in report_text


def test_validator_detects_csv_comma_formatting_error(tmp_path):
    matching_tsv = str(tmp_path / "matching_results.tsv")
    candidate_tsv = str(tmp_path / "candidate_pairs.tsv")

    # Write CSV with commas instead of TSV with tabs
    with open(matching_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id,matched_entity_ids\n")
        f.write("S1-100,S2-200\n")

    with open(candidate_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-100\tS2-200\n")

    result = SubmissionValidator.validate_submission(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        expected_s1_ids={"S1-100"},
    )

    assert result.is_valid is False
    assert result.criteria_checks["correct_tsv_format"] is False
    assert any("Header contains commas but NO tab character" in e for e in result.errors)


def test_validator_detects_corrupt_empty_strings(tmp_path):
    matching_tsv = str(tmp_path / "matching_results.tsv")
    candidate_tsv = str(tmp_path / "candidate_pairs.tsv")

    # Write corrupt representation "None" or "NaN" for singleton
    with open(matching_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-100\tNone\n")

    with open(candidate_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-100\t\n")

    result = SubmissionValidator.validate_submission(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        expected_s1_ids={"S1-100"},
    )

    assert result.is_valid is False
    assert result.criteria_checks["empty_lists_correctly_represented"] is False
    assert any("Corrupt empty string representation" in e for e in result.errors)


def test_validator_detects_duplicate_s1_and_intra_list_duplicates(tmp_path):
    matching_tsv = str(tmp_path / "matching_results.tsv")
    candidate_tsv = str(tmp_path / "candidate_pairs.tsv")

    with open(matching_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-100\tS2-200,S2-200\n")  # Intra-list duplicate
        f.write("S1-100\tS2-201\n")          # Duplicate S1 row

    with open(candidate_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-100\tS2-200\n")

    result = SubmissionValidator.validate_submission(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        expected_s1_ids={"S1-100"},
    )

    assert result.is_valid is False
    assert result.criteria_checks["no_duplicate_s1_rows"] is False
    assert result.criteria_checks["no_duplicate_ids_in_lists"] is False


def test_validator_detects_self_matches_and_invalid_prefixes(tmp_path):
    matching_tsv = str(tmp_path / "matching_results.tsv")
    candidate_tsv = str(tmp_path / "candidate_pairs.tsv")

    with open(matching_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-100\tS1-999,INVALID-101\n")  # S1 self match and bad prefix

    with open(candidate_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-100\tS1-999,INVALID-101\n")

    result = SubmissionValidator.validate_submission(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        expected_s1_ids={"S1-100"},
    )

    assert result.is_valid is False
    assert result.criteria_checks["only_s2_s3_matches"] is False
    assert any("self-matches" in e for e in result.errors)
    assert any("without S2-/S3- prefix" in e for e in result.errors)


def test_validator_detects_match_not_in_candidates(tmp_path):
    matching_tsv = str(tmp_path / "matching_results.tsv")
    candidate_tsv = str(tmp_path / "candidate_pairs.tsv")

    with open(matching_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-100\tS2-500\n")  # Match S2-500 is NOT in candidates

    with open(candidate_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("S1-100\tS2-200,S3-300\n")

    result = SubmissionValidator.validate_submission(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        expected_s1_ids={"S1-100"},
    )

    assert result.is_valid is False
    assert result.criteria_checks["matches_subset_of_candidates"] is False
    assert any("matches are NOT in candidates" in e for e in result.errors)


def test_submission_writer_cleans_and_enforces_subset(tmp_path):
    matching_tsv = str(tmp_path / "matching_results.tsv")
    candidate_tsv = str(tmp_path / "candidate_pairs.tsv")

    writer = SubmissionWriter(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        enforce_subset=True,
    )
    with writer:
        # Pass duplicate targets, self match S1-999, and a match not initially in candidates
        writer.write_entity_result(
            s1_id="S1-100",
            matched_ids=["S2-10", "S2-10", "S1-999", "S3-30"],
            candidate_ids=["S2-10", "S2-20"],
        )

    # Validate output written by SubmissionWriter
    result = SubmissionValidator.validate_submission(
        matching_path=matching_tsv,
        candidate_path=candidate_tsv,
        expected_s1_ids={"S1-100"},
    )

    assert result.is_valid is True
    assert result.criteria_checks["matches_subset_of_candidates"] is True
    assert result.criteria_checks["only_s2_s3_matches"] is True
    assert result.criteria_checks["no_duplicate_ids_in_lists"] is True
