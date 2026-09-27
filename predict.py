"""Entrypoint script for running test inference and generating submission files.

Usage:
    python predict.py
    python predict.py --sample-size 5000
    python predict.py --threshold 0.88
"""

import sys
from src.business_entity_resolution.inference.pipeline import run_cli_inference

if __name__ == "__main__":
    run_cli_inference()
