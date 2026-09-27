"""Prepares and verifies the official competition submission package.

Verifies:
- output/
    matching_results.tsv
    candidate_pairs.tsv
- code/business_entity_resolution/
    src/
    README.md
    requirements.txt
- Documentation_template.md
"""

import os
import shutil
import zipfile
import subprocess
import sys


def clean_pycache(root_dir: str):
    """Recursively removes all __pycache__ folders and .pyc files."""
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for d in dirs:
            if d in ("__pycache__", ".pytest_cache", "__MACOSX"):
                p = os.path.join(root, d)
                shutil.rmtree(p, ignore_errors=True)
        for f in files:
            if f.endswith((".pyc", ".pyo", ".DS_Store", "Thumbs.db")):
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass


def prepare_package():
    base_dir = os.path.abspath(".")
    print("=" * 80)
    print("PREPARING FINAL COMPETITION SUBMISSION PACKAGE")
    print("=" * 80)

    # 1. Clean caches from workspace
    print("\n1. Cleaning cache and temporary files...")
    clean_pycache(base_dir)

    # 2. Ensure output/ directory and TSV files
    print("\n2. Setting up output/ directory...")
    output_dir = os.path.join(base_dir, "output")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
        src_f = os.path.join(outputs_dir, fname)
        dst_f = os.path.join(output_dir, fname)
        if os.path.isfile(src_f):
            shutil.copy2(src_f, dst_f)
            print(f"  [OK] output/{fname} ({os.path.getsize(dst_f):,} bytes)")
        elif os.path.isfile(dst_f):
            print(f"  [OK] output/{fname} ({os.path.getsize(dst_f):,} bytes)")
        else:
            raise FileNotFoundError(f"Missing required output file: {fname}")

    # 3. Setup code/business_entity_resolution/
    print("\n3. Setting up code/business_entity_resolution/...")
    code_pkg_dir = os.path.join(base_dir, "code", "business_entity_resolution")
    os.makedirs(code_pkg_dir, exist_ok=True)

    # Copy src/ directory
    dst_src = os.path.join(code_pkg_dir, "src")
    if os.path.exists(dst_src):
        shutil.rmtree(dst_src)
    shutil.copytree(os.path.join(base_dir, "src"), dst_src)
    clean_pycache(dst_src)
    print("  [OK] code/business_entity_resolution/src/")

    # Copy key files
    files_to_copy = [
        "README.md",
        "requirements.txt",
        "config.yaml",
        "run_pipeline.py",
        "build_train_dataset.py",
        "predict.py",
        "validate_submission.py",
        "run_experiments.py",
    ]
    for fname in files_to_copy:
        src_path = os.path.join(base_dir, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(code_pkg_dir, fname))
            print(f"  [OK] code/business_entity_resolution/{fname}")

    # Copy models/ if present
    models_src = os.path.join(base_dir, "models")
    models_dst = os.path.join(code_pkg_dir, "models")
    if os.path.isdir(models_src):
        if os.path.exists(models_dst):
            shutil.rmtree(models_dst)
        shutil.copytree(models_src, models_dst)
        print("  [OK] code/business_entity_resolution/models/")

    # 4. Verify Documentation_template.md
    print("\n4. Verifying Documentation_template.md...")
    doc_path = os.path.join(base_dir, "Documentation_template.md")
    if not os.path.isfile(doc_path):
        raise FileNotFoundError("Documentation_template.md is missing at project root!")
    print(f"  [OK] Documentation_template.md ({os.path.getsize(doc_path):,} bytes)")

    # 5. Build official submission.zip package
    print("\n5. Building submission.zip archive...")
    zip_path = os.path.join(base_dir, "submission.zip")
    if os.path.isfile(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = os.path.join(output_dir, fname)
            zf.write(fpath, arcname=f"output/{fname}")

        # Add code files
        for root, _, files in os.walk(code_pkg_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                zf.write(full_path, arcname=rel_path.replace("\\", "/"))

        # Add Documentation_template.md
        zf.write(doc_path, arcname="Documentation_template.md")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  [OK] Created submission.zip ({zip_size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("SUBMISSION PACKAGE VERIFICATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    prepare_package()
