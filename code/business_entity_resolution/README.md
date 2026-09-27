# Amazon ML Challenge 2026: Business Entity Resolution

A modular, reproducible, high-precision Machine Learning application for multi-source Business Entity Resolution.

---

## 1. Problem Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources:
* **Source 1:** Deduplicated reference source (`S1-` prefix).
* **Source 2:** Noisy business records (`S2-` prefix).
* **Source 3:** Noisy business records (`S3-` prefix).

For every Source 1 business entity, the objective is to identify all matching records from Source 2 and Source 3 while minimizing false merges.

### Key Rules & Constraints:
* **Metric:** Macro $F_{0.5}$ (precision-weighted, penalizing false merges $4\times$ more heavily than false negatives).
* **Singletons:** Exactly 5.58% of entities in the training data have no matches. A singleton correctly predicted as empty yields $1.0$, while any false link yields $0.0$.
* **Open-Set Countries:** The test data introduces **France** (~15% of records) in addition to **US** and **India**. The pipeline dynamically supports arbitrary open-set country labels without hardcoding.
* **Format:** Strict tab-separated (`.tsv`) files with headers `source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`.

---

## 2. Project Directory Structure

```text
business-entity-resolution/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── run_pipeline.py
├── train.py
├── evaluate.py
├── predict.py
│
├── data/
│   ├── train/                 # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                  # test_source1.tsv, test_source2.tsv, test_source3.tsv
│
├── models/                    # Serialized model artifacts & optimal thresholds
├── outputs/                   # matching_results.tsv, candidate_pairs.tsv, validation reports
│
├── src/
│   └── business_entity_resolution/
│       ├── data/
│       │   ├── loader.py      # TSV reader, streaming chunks, raw preservation
│       │   └── validator.py   # Schema, ID prefix, duplicate, and output validator
│       ├── preprocessing/     # Unicode, abbreviations, legal suffixes, tokenizers
│       ├── blocking/          # Multi-strategy candidate generation & indexing
│       ├── features/          # Pairwise string & token similarity extractors
│       ├── models/            # Classifiers & F0.5 threshold optimizer
│       ├── evaluation/        # Macro F0.5 metric computation
│       ├── inference/         # Country-partitioned streaming inference pipeline
│       └── utils/             # Config loader, logger
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_validator.py
│   └── test_dataset_integration.py
│
└── docs/
    ├── data_profile.md        # Comprehensive dataset profile report
    ├── methodology.md         # Architecture and modeling writeup
    └── experiments.md         # Experiment tracking log
```

---

## 3. Data Loading & Validation Layer

The data-loading and validation layer (`src/business_entity_resolution/data/`) ensures safe, memory-efficient, and schema-compliant access to datasets.

### Key Capabilities:
1. **Explicit Delimiter Handling:** Strictly reads `.tsv` files using `sep="\t"` to avoid comma collision inside business addresses and ID lists.
2. **Missing Value Safety:** Safely preserves missing values as empty strings `""` without corrupting pandas types, keeping address omissions in S2/S3 (~3%) distinguishable.
3. **Preservation of Raw Inputs:** Stores original values in `raw_business_name`, `raw_business_address`, and `raw_country` before normalization.
4. **Streaming / Chunking Support:** Iterates through millions of rows in configurable chunk sizes to maintain a sub-1GB memory footprint.
5. **Open-Set Country Support:** Dynamically processes any country label without hardcoded filters.
6. **Strict Schema & Prefix Validation:** Verifies required columns, detects ID prefix violations (`S1-`, `S2-`, `S3-`), identifies duplicate IDs, and prevents self-matches.

---

## 4. Usage Examples

### 4.1 Loading Source Records with Validation
```python
from src.business_entity_resolution.data.loader import DataLoader

# Initialize loader with data directory
loader = DataLoader("data/train")

# Load Source 1 records with schema & prefix validation
df_s1 = loader.load_source_df(
    filename="train_source1.tsv",
    nrows=10000,
    validate=True,
    expected_prefix="S1-",
    preserve_raw=True,
)

print(df_s1.head())
# Access preserved raw fields: df_s1['raw_business_name']
```

### 4.2 Streaming Millions of Records in Chunks
```python
from src.business_entity_resolution.data.loader import DataLoader

loader = DataLoader("data/train")

# Stream large files without exhausting RAM
for chunk in loader.stream_source_chunks("train_source2.tsv", chunk_size=50000):
    print(f"Loaded chunk with {len(chunk)} records")
    # Process batch...
```

### 4.3 Loading and Validating Ground Truth
```python
from src.business_entity_resolution.data.loader import DataLoader

loader = DataLoader("data/train")

# Returns mapping: { 'S1-00001': {'S2-00047', 'S3-00812'}, 'S1-00002': set() }
ground_truth_map = loader.load_ground_truth(
    filename="train_ground_truth.tsv",
    validate=True,
)
```

### 4.4 Validating DataFrames & Submissions
```python
from src.business_entity_resolution.data.validator import DataValidator, OutputValidator

# Validate an in-memory DataFrame
is_valid, errors, warnings = DataValidator.validate_source_dataframe(
    df=df_s1,
    expected_prefix="S1-",
    file_label="train_source1",
)

# Validate competition output TSVs before submission
report = OutputValidator.validate_submission(
    matching_path="outputs/matching_results.tsv",
    candidate_path="outputs/candidate_pairs.tsv",
    test_dir="data/test",
    check_target_ids=False,
)

if report.is_valid:
    print("Submission PASSED all format and consistency checks!")
else:
    print("Submission errors:", report.errors)
```

---

## 5. Preprocessing Layer

The preprocessing layer (`src/business_entity_resolution/preprocessing/`) provides conservative, jurisdiction-aware normalization while always retaining original inputs.

### 5.1 Business Name Normalization
```python
from src.business_entity_resolution.preprocessing.name_normalizer import BusinessNameNormalizer

normalizer = BusinessNameNormalizer()
result = normalizer.normalize("Thermal & Fils SASU")

print("Original:       ", result.original)            # 'Thermal & Fils SASU'
print("Normalized:     ", result.normalized)          # 'thermal and fils sasu'
print("Without Legal:  ", result.clean_without_legal)  # 'thermal and fils'
print("Legal Suffix:   ", result.detected_legal_suffix) # 'sasu'
print("Tokens:         ", result.tokens)              # ['thermal', 'and', 'fils', 'sasu']
```

### 5.2 Address Normalization
```python
from src.business_entity_resolution.preprocessing.address_normalizer import AddressNormalizer

addr_normalizer = AddressNormalizer()
result = addr_normalizer.normalize("AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203")

print("Original:   ", result.original)
print("Normalized: ", result.normalized)
# Standardized numbers without leading zeroes:
print("Numbers:    ", result.numeric_tokens)  # ['684', '989', '9487203']
print("Empty?      ", result.is_empty)        # False
```

---

## 6. Pairwise Feature Engineering

The feature engineering layer (`src/business_entity_resolution/features/`) generates 36 explainable pairwise similarity features across business names, addresses, and country labels.

### 6.1 Extracted Feature Groups

* **Name Features:** Exact normalized match, exact clean match without legal suffix, Levenshtein edit similarity, token Jaccard similarity, shared token count and ratio, character 3-gram Jaccard, token sort ratio, TF-IDF cosine similarity, absolute and relative length differences, legal suffix equality, and missing name indicator.
* **Address Features:** Exact match, Levenshtein edit similarity, token Jaccard similarity, shared token count and ratio, numeric token Jaccard and count (matching house numbers and pincodes), exact numeric match, token sort ratio, TF-IDF cosine similarity, length differences, and missing address indicators (`addr_is_missing_either`, `addr_is_missing_both`).
* **Country Features:** Open-set exact country equality (`country_exact_match`), missing country indicator.
* **Structural & Composite Features:** Target source prefix indicator (`target_is_source2`), length statistics, token count difference, and composite heuristic score.

### 6.2 Feature Builder Usage
```python
from src.business_entity_resolution.features.feature_builder import FeatureBuilder

builder = FeatureBuilder()
# Fit TF-IDF on training reference names/addresses to prevent data leakage
builder.fit_reference_tfidf(
    s1_names=["Maure Williams Colombier Inc", "Dahlia Power Reliable Scientific LLC"],
    s1_addresses=["85 Wayne Avenue, NY", "630 45th Terrace, MO"]
)

# Extract feature dictionary for a candidate pair
features = builder.extract_pair_dict(s1_record, target_record)
print("Levenshtein Name Similarity: ", features["name_levenshtein_sim"])
print("Numeric Token Overlap:       ", features["addr_numeric_overlap_count"])
print("Composite Match Score:       ", features["composite_heuristic_score"])
```

---

## 7. Final Inference Pipeline

The production inference pipeline (`src/business_entity_resolution/inference/pipeline.py` & `predict.py`) executes end-to-end test scoring compliant with all challenge criteria:
1. Loads test Source 1, Source 2, and Source 3 records.
2. Normalizes entity names and addresses (extracting legal suffixes, numeric street numbers, unicode normalization).
3. Generates high-recall candidates via country-isolated multi-strategy blocking.
4. Generates 36 explainable pairwise similarity features.
5. Loads the winning gradient boosting model (`models/best_model.joblib`).
6. Applies the precision-optimal decision threshold ($\tau = 0.88$).
7. Generates `outputs/matching_results.tsv` and `outputs/candidate_pairs.tsv`.
8. Enforces exactly one row per test Source 1 entity, preserves empty match lists for singletons, guarantees matches are subsets of candidates, and validates compliance using `dataset/student_resource/utils/validate_submission.py`.

### 7.1 Running Test Predictions
```bash
# Run test inference on a sample
python predict.py --sample-size 5000

# Run test inference with custom model or threshold override
python predict.py --model-path models/best_model.joblib --threshold 0.88

### 7.2 Strict Submission Validation
The strict submission validator (`src/business_entity_resolution/data/submission_validator.py` and `validate_submission.py`) exhaustively checks all 11 competition rules:
* Correct TSV format (tab-separated, no delimiter corruption).
* Exact column headers (`source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`).
* Every Source 1 test entity exists and is present.
* Exactly one row per Source 1 entity (no extra, no missing).
* No duplicate Source 1 rows.
* Only S2/S3 IDs appear as matches (zero S1 self-matches).
* All IDs exist in test data (verifying both S1 and S2/S3 IDs against test sources).
* No duplicate IDs inside comma-separated lists.
* Candidate IDs valid and compliant.
* Every final match strictly appears in candidate_pairs.tsv.
* Empty lists are correctly represented for singletons (`s1_id\t\n`).

```bash
# Run strict validation
python validate_submission.py

# Run strict validation with ID existence checking across ~10M test target entities
python validate_submission.py --check-target-ids
```

---

## 8. FastAPI Backend API

A production-grade REST API backend (`app/backend/`) wraps the machine learning pipeline, decoupling business logic in `src/business_entity_resolution/services/ml_service.py` from route orchestration.

### 8.1 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status, active model, optimal threshold, dataset availability |
| `GET` | `/dataset/stats` | Train/test file sizes, country distribution, singleton & missing rates |
| `POST` | `/train` | Train matching model (Logistic Regression or Gradient Boosting) |
| `POST` | `/evaluate` | Macro $F_{0.5}$ evaluation & optional decision threshold sweep |
| `POST` | `/predict` | Run test inference pipeline generating TSVs |
| `POST` | `/validate` | Strictly validate submission TSV files across all 11 criteria |
| `GET` | `/metrics` | Baseline evaluation metrics, confusion matrix, and threshold curves |
| `GET` | `/experiments` | Controlled experiment results (Experiments A through F) |
| `GET` | `/matches/{source1_entity_id}` | Look up entity record, matches, candidates, and resolved target details |

### 8.2 Launching the Server
```bash
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## 9. Running Tests

To run the complete test suite:

```bash
python -m pytest tests/ -v
```

All 82 unit and integration tests execute in ~9.5 seconds.
