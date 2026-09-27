# Amazon ML Challenge 2026: Comprehensive Data Profile Report

**Document:** Data Inspection & Profiling Analysis  
**Repository:** `business-entity-resolution`  
**Dataset Scope:** 7 TSV files (~26.4 million records total)  
**Inspection Date:** 2026-09-26  

---

## 1. Executive Summary & File Inventory

The dataset consists of **7 tab-separated (`.tsv`) files** organized into training and test splits. All files were systematically inspected, verified for schema compliance, and profiled for data distributions, missing values, duplicates, and noise patterns.

| File Path | Role | File Size | Total Data Rows | Header / Delimiter |
| :--- | :--- | :--- | :--- | :--- |
| `data/train/train_source1.tsv` | Deduplicated Reference Source | 200.34 MB | **2,206,821** | Strict Tab (`\t`) |
| `data/train/train_source2.tsv` | Noisy Business Records | 466.63 MB | **5,034,616** | Strict Tab (`\t`) |
| `data/train/train_source3.tsv` | Noisy Business Records | 480.37 MB | **5,285,603** | Strict Tab (`\t`) |
| `data/train/train_ground_truth.tsv` | Reference Matching Labels | 121.13 MB | **2,206,821** | Strict Tab (`\t`) |
| `data/test/test_source1.tsv` | Test Reference Source | 166.91 MB | **1,732,544** | Strict Tab (`\t`) |
| `data/test/test_source2.tsv` | Test Target Candidates | 485.86 MB | **4,887,273** | Strict Tab (`\t`) |
| `data/test/test_source3.tsv` | Test Target Candidates | 482.56 MB | **5,082,316** | Strict Tab (`\t`) |
| **Total** | **All 7 Files** | **2.39 GB** | **26,435,994** | **100% TSV compliant** |

> [!NOTE]
> All files use explicit tab delimiters (`\t`). Commas are strictly reserved for internal address parts and list separation in ground truth. No quoting is used.

---

## 2. Schema & Column Verification

### 2.1 Source Files Schema
All 6 source files (`train_source1`, `train_source2`, `train_source3`, `test_source1`, `test_source2`, `test_source3`) adhere to the exact 4-column schema:

```
['entity_id', 'business_name', 'business_address', 'country']
```

- **Actual columns found:** Exact match across all 6 files.
- **Extra/Unexpected columns:** None.
- **Delimiter:** Strict Tab (`\t`). No comma-separated header errors.

### 2.2 Ground Truth Schema
The training ground truth file (`train_ground_truth.tsv`) adheres to the exact 2-column schema:

```
['source1_entity_id', 'matched_entity_ids']
```

- **Exact 1-to-1 correspondence:** Matches the 2,206,821 entities in `train_source1.tsv`.
- **Delimiter:** Strict Tab (`\t`). Matched IDs are comma-separated within the second column.

---

## 3. Entity ID Prefix & Uniqueness Verification

Entity ID prefixes indicate the source of origin and must strictly match the rules:
- Source 1 IDs: Must begin with `S1-`
- Source 2 IDs: Must begin with `S2-`
- Source 3 IDs: Must begin with `S3-`

### Inspection Results:
* **Prefix Violations:** **0 violations** found across all 26,435,994 records (100.00% valid).
* **Duplicate Entity IDs:** **0 duplicates** found in any file. Every entity ID is globally unique within its respective file.
* **Ground Truth ID Verification:**
  * 0 self-matches (no `S1-` IDs appear in `matched_entity_ids`).
  * 0 malformed IDs (all matched IDs begin with either `S2-` or `S3-`).

---

## 4. Missing Value Analysis

Every column across all files was audited for missing, empty, or whitespace-only values:

| File | `entity_id` Missing | `business_name` Missing | `business_address` Missing | `country` Missing |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 0 (0.00%) | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| `train_source2.tsv` | 0 (0.00%) | 0 (0.00%) | **168,967 (3.36%)** | 0 (0.00%) |
| `train_source3.tsv` | 0 (0.00%) | 0 (0.00%) | **175,916 (3.33%)** | 0 (0.00%) |
| `test_source1.tsv` | 0 (0.00%) | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| `test_source2.tsv` | 0 (0.00%) | 0 (0.00%) | **129,408 (2.65%)** | 0 (0.00%) |
| `test_source3.tsv` | 0 (0.00%) | 0 (0.00%) | **136,098 (2.68%)** | 0 (0.00%) |

> [!IMPORTANT]
> **Key Data Insight:**
> 1. `entity_id`, `business_name`, and `country` are **100% populated** across all 26.4 million records.
> 2. `business_address` is **100% populated in Source 1** (both train and test).
> 3. However, **~2.7% to 3.4% of records in Source 2 and Source 3 have missing addresses** (represented as empty string `""` or whitespace).
> 4. **Pipeline Impact:** Feature engineering must gracefully handle empty addresses using missing-value indicators and fall back on robust name similarity metrics when address comparisons are impossible.

---

## 5. Country Distribution & Open-Set Analysis

The competition rules require treating `country` as an **OPEN SET**. Profiling confirmed this critical property:

### 5.1 Training Set Country Distribution
The training set exclusively covers 2 countries:

| Country | `train_source1` | `train_source2` | `train_source3` |
| :--- | :--- | :--- | :--- |
| **US** | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.98%) |
| **India** | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.02%) |

### 5.2 Test Set Country Distribution
The test set introduces **France** as an unobserved country during training, and shifts the relative country proportions:

| Country | `test_source1` | `test_source2` | `test_source3` |
| :--- | :--- | :--- | :--- |
| **India** | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) |
| **US** | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) |
| **France** | **259,452 (14.98%)** | **703,378 (14.39%)** | **731,615 (14.40%)** |

> [!WARNING]
> **Critical Architectural Constraint:**
> - France constitutes **~15% of the test set** (259,452 Source 1 records and ~1.43 million Source 2/3 records).
> - Models and blocking strategies must **never** hardcode `{"US", "India"}` or use one-hot encoding for countries.
> - Blocking and candidate generation should partition dynamically by `record["country"]` or evaluate string equality (`country1 == country2`).
> - Text normalization must support French legal suffixes (e.g., `SARL`, `SAS`, `SASU`, `EURL`, `SCI`) and address tokens (`Rue`, `Boulevard`, `Impasse`, `Avenue`).

---

## 6. Ground Truth Analysis & Singleton Dynamics

Analysis of all 2,206,821 labels in `train_ground_truth.tsv` revealed the following characteristics:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Source 1 Entities** | 2,206,821 | Exactly matches `train_source1.tsv` rows |
| **Singletons (0 matches)** | **123,247 (5.58%)** | Empty `matched_entity_ids` field |
| **Entities with $\ge 1$ match** | **2,083,574 (94.42%)** | At least one valid S2 or S3 link |
| **Total Matched Links** | **7,638,365** | Average of 3.46 links per S1 overall |
| **Source 2 Links** | 3,693,619 (48.36%) | Balanced link distribution |
| **Source 3 Links** | 3,944,746 (51.64%) | Balanced link distribution |

### Match Count Distribution (for Non-Singletons):
- **Min:** 1 match
- **Max:** 11 matches
- **Mean:** 3.67 matches
- **Median:** 4.0 matches
- **Standard Deviation:** 1.53
- **25th Percentile:** 3.0 matches
- **75th Percentile:** 5.0 matches

### Singleton Evaluation Impact:
Under macro $F_{0.5}$:
- A true singleton predicted as empty yields **$F_{0.5} = 1.0$**.
- A true singleton falsely predicted with any match yields **$F_{0.5} = 0.0$**.
- Therefore, threshold optimization is vital to suppress false positive links on the 5.6% singleton population.

---

## 7. Concrete Examples & Noise Patterns

### 7.1 Business Name Noise Patterns
Inspecting real ground-truth pairs highlighted key noise dimensions:

1. **Typographical Errors & OCR Noise:**
   - S1: `Maure Williams Colombier Inc`
   - S2: `Maure Wilblims Colombier Inc` (`Wilblims` vs `Williams`)
   - S3: `Dahlia Ponr Reliable Scientific LLC` (`Ponr` vs `Power`)
2. **Legal Suffix Variations:**
   - S1: `Maure Williams Colombier Inc`
   - S2: `Maure Williams Colombier` (Suffix dropped)
   - S3: `Maure Williams Inc Center` (Suffix rearranged)
3. **Transliteration (Indian Languages):**
   - S1: `Ss Food Private Limited`
   - S2: `एसएस फूड प्राइवेट लिमिटेड` (Hindi Devanagari script transliteration)
   - S1: `Raj Investments LLP`
   - S2: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (Tamil script transliteration)
4. **Domain Name Formats:**
   - S1: `Maure Williams Colombier Inc`
   - S3: `maurewilliamscolombier.com` (URL format)
5. **French Business Names:**
   - S1: `Thermal & Fils SASU` | S1: `Elephant Centre EURL` | S1: `ZNB Club SARL`
   - S2: `sci ligue ici parents` | S2: `OZT ÀMICALE SAS` | S2: `QHC Culture [EURL]`

### 7.2 Business Address Noise Patterns
1. **Component Reordering:**
   - S1: `630 45th Terrace, Kansas City, MO`
   - S2: `KANSAS CITY, MO, 630 45ND TERRACE, null` (City/state moved to front)
   - S3: `Missouri, 630 45th Terrace, Kansas City` (State first)
2. **Numeric Token Alignment (Leading Zeros & Abbreviations):**
   - S1: `Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh`
   - S2: `AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश`
   - *Key observation:* Normalizing numbers (`0684` $\to$ `684`) links transliterated entities with 100% precision.
3. **Landmark-based & Local References:**
   - India: `Opp. RTA Office`, `B/H Natraj Cinema`, `Near Mother India Public School`, `Flat No.207`
   - US: `Unit APARTMENT G`, `Suite 300`, `Building 4`
   - France: `5 bis Rue Pierre Dignac`, `NO. 5 ALLÉE DES HÊTRES`, `175 Boulevard du Président Franklin Roosevelt`
4. **Missing Addresses:**
   - `S2-681193310` has empty address `""` while business name matches with minor typo.

---

## 8. Summary of Findings vs. Expected Schema

| Aspect | Expected Specification | Actual Finding in Data | Status / Action Needed |
| :--- | :--- | :--- | :--- |
| **File Format** | Tab-separated `.tsv` | 100% tab-separated, no comma delimiters | Compliant |
| **Columns** | 4 columns for source, 2 for GT | Exact column match across all files | Compliant |
| **Entity Prefixes** | `S1-`, `S2-`, `S3-` | 0 prefix violations across 26.4M rows | Compliant |
| **Uniqueness** | Unique IDs per file | 0 duplicate entity IDs | Compliant |
| **Countries** | Open-set (US, India, France) | US & India in train; US, India, & France in test | Requires dynamic country partitioning |
| **Missing Values** | Missing fields expected | Address is empty in ~3% of S2/S3 records; Name & Country 100% populated | Address empty indicator + robust name fallbacks |
| **Singletons** | S1 entities with 0 matches | Exactly 5.58% singletons in ground truth | Threshold optimization required for precision |
| **Multi-matches** | Multiple matches per S1 | Non-singletons have 1 to 11 matches (mean 3.67) | Classifier must rank & multi-predict |

---
*Report generated and validated autonomously by Antigravity IDE.*
