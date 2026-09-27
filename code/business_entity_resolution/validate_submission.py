"""CLI entrypoint for strict submission validation.

Usage:
    python validate_submission.py
    python validate_submission.py --matching outputs/matching_results.tsv --candidate outputs/candidate_pairs.tsv
    python validate_submission.py --check-target-ids
"""

from src.business_entity_resolution.data.submission_validator import main

if __name__ == "__main__":
    main()
