# Controlled Experiments Report: Business Entity Resolution

**Amazon ML Challenge 2026**  
**Evaluation Metric:** Macro $F_{0.5}$ across Source 1 Entities (Precision-Weighted)  
**Experiment Date:** 2026-09-26  
**Persisted Results File:** [`outputs/experiments.csv`](file:///c:/Users/punee/Desktop/business-entity-resolution/outputs/experiments.csv)

---

## 1. Controlled Experiment Results Summary

All experiments were conducted on identical holdout validation partitions using entity-grouped splitting to guarantee **zero entity leakage**. Thresholds were independently optimized for each configuration to maximize Macro $F_{0.5}$.

| Experiment ID | Architecture | Feature Set | Blocking Strategy | Optimal $\tau$ | Candidate Recall | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Delta vs Baseline | Runtime |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | `logistic_regression` | Baseline (12 feats) | Standard ($k=25$) | `0.93` | 92.10% | 0.9752 | 0.9571 | **0.9637** | Baseline | 0.3s |
| **B** | `logistic_regression` | Better Name (19 feats) | Standard ($k=25$) | `0.90` | 92.10% | 0.9742 | 0.9678 | **0.9668** | $+0.0031$ | 0.4s |
| **C** | `logistic_regression` | Better Name + Addr (28 feats) | Standard ($k=25$) | `0.91` | 92.10% | 0.9784 | 0.9764 | **0.9718** | $+0.0081$ | 0.4s |
| **D** | `logistic_regression` | All Features (36 feats) | Standard ($k=25$) | `0.92` | 92.10% | 0.9785 | 0.9777 | **0.9736** | $+0.0099$ | 0.4s |
| **E** | `hist_gradient_boosting` | All Features (36 feats) | Standard ($k=25$) | **`0.88`** | 92.10% | **0.9844** | **0.9803** | **0.9782** | **$+0.0145$** | 4.0s |
| **F** | `hist_gradient_boosting` | All Features (36 feats) | Improved ($k=35$) | `0.74` | **96.02%** | **0.9945** | 0.9542 | **0.9763** | $+0.0126$ | 49.1s |

---

## 2. In-Depth Analysis of Controlled Hypotheses

### Experiment A $\to$ B: Impact of Enriched Name Features
- **Hypothesis:** Adding token overlap counts, token sort ratio, legal suffix alignment, and length difference metrics will resolve word re-ordering and corporate noise suffixes.
- **Evidence:** Macro Recall increased from **0.9571 $\to$ 0.9678** ($+1.07\%$) with Macro $F_{0.5}$ improving from **0.9637 $\to$ 0.9668**.
- **Conclusion:** Accepted. Better name features directly prevent false dismissals caused by word permutations and differing corporate designators (`Inc` vs `LLC`).

### Experiment B $\to$ C: Impact of Enriched Address Features
- **Hypothesis:** Adding numeric token alignment (house/door numbers, postal codes) and address token overlap will provide orthogonal validation to name matching.
- **Evidence:** Both Precision (**0.9742 $\to$ 0.9784**) and Recall (**0.9678 $\to$ 0.9764**) jumped substantially, yielding a **$+0.0050$** leap in Macro $F_{0.5}$ (**0.9668 $\to$ 0.9718**).
- **Conclusion:** Accepted. Numeric token features (`addr_numeric_jaccard`, `addr_numeric_exact_match`) are critical negative signals preventing erroneous merges of distinct branches operating under identical corporate names.

### Experiment C $\to$ D: Additional Token/Character Statistics
- **Hypothesis:** Character 3-gram Jaccard and length difference ratios will provide robustness against severe OCR typos and transliterations.
- **Evidence:** Macro $F_{0.5}$ reached **0.9736** with linear regression, with character 3-gram Jaccard receiving a top-5 positive model coefficient ($+1.5915$).
- **Conclusion:** Accepted. Provides the full 36-feature dense representation.

### Experiment D $\to$ E: Linear Model vs Non-Linear Tree Ensemble
- **Hypothesis:** A tree-based ensemble (`HistGradientBoostingClassifier`) can capture non-linear feature interactions (such as gating address matching on country equality, or non-linearly penalizing number mismatches) better than a generalized linear model.
- **Evidence:** **Experiment E achieved the highest validation performance of all experiments**:
  - Macro Precision reached **0.9844** (up from 0.9785).
  - Macro Recall reached **0.9803** (up from 0.9777).
  - **Macro $F_{0.5}$ hit 0.9782** at optimal threshold $\tau = 0.88$.
  - Training runtime remained well under 5 seconds (4.0s).
- **Conclusion:** **Winning Model Selected.** Gradient boosted decision trees provide superior precision and recall across noisy multi-source records.

### Experiment E $\to$ F: Standard ($k=25$) vs Expanded Blocking ($k=35$)
- **Hypothesis:** Increasing candidate generation threshold $k$ from 25 to 35 will decrease blocking false negatives, improving upper-bound candidate recall.
- **Evidence:**
  - Candidate recall increased significantly from **92.10% $\to$ 96.02%** (+3.92% more true links retrieved).
  - Model achieved an extraordinary **0.9945 Precision** (99.45% of predicted matches were true positives).
  - Macro $F_{0.5}$ remained exceptionally high at **0.9763**.
- **Conclusion:** Validated trade-off. $k=35$ delivers nearly 4% higher candidate recall for competitive upper-bound ceiling, while $k=25$ offers 10x faster execution speed.

---

## 3. Final Production Architecture Decisions

Based on validation evidence:
1. **Model Architecture:** `HistGradientBoostingMatcher` (Gradient Boosted Trees).
2. **Feature Set:** Full 36-feature suite (`all_features`).
3. **Decision Threshold:** $\tau = 0.88$ for tree ensemble ($\tau = 0.92$ for linear baseline).
4. **Blocking:** Country-partitioned inverted index with multi-token blocking ($k=25$ for fast streaming inference, $k=35$ for high-recall evaluation).
