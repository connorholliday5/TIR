# TIR coding benchmark

What the models are worth in coder time. Regenerate with `python -m src.report_benchmark`.

Held-out split: **13,644 rows**, never seen during training.

Accuracy alone is a poor guide here — the categories are very unbalanced, so a model can score well while getting the whole rare tail wrong. Macro-F1 and the coverage curve below are the numbers that matter.

## Accuracy by field

| Field | Rows judged | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| metric_cat | 13,644 | 0.9283 | 0.6053 |
| process_cat | 9,235 | 0.8823 | 0.7433 |
| process_sub | 9,157 | 0.8083 | 0.6304 |
| process_l3 | 8,247 | 0.7357 | 0.5401 |

> Read each figure against how consistently people code that field — see `data_sufficiency.md`. A model matching its coders is at the ceiling, not failing.

## Coverage at a precision target

The number that converts to time saved: at a given confidence threshold, what share of TIRs can be coded automatically, and how often is that automatic answer right?

### metric_cat

| Threshold | Auto-coded | Precision | Left for a coder |
| ---: | ---: | ---: | ---: |
| 0.00 | 100.0% | 92.8% | 0.0% |
| 0.50 | 98.8% | 93.4% | 1.2% |
| 0.60 | 95.4% | 94.7% | 4.6% |
| 0.70 | 91.4% | 96.0% | 8.6% |
| 0.80 ← configured | 86.1% | 97.3% | 13.9% |
| 0.90 | 75.9% | 98.5% | 24.1% |
| 0.95 | 62.1% | 99.2% | 37.9% |

- **95% precision** is reached at threshold 0.63, coding **94.3%** of TIRs automatically (actual precision 95.1%).
- **90% precision** is reached at threshold 0.05, coding **100.0%** of TIRs automatically (actual precision 92.8%).

Currently configured `review_threshold`: **0.80**.

### process_cat

| Threshold | Auto-coded | Precision | Left for a coder |
| ---: | ---: | ---: | ---: |
| 0.00 | 100.0% | 88.2% | 0.0% |
| 0.50 | 93.2% | 91.5% | 6.8% |
| 0.60 | 86.4% | 93.6% | 13.6% |
| 0.70 | 79.6% | 95.3% | 20.4% |
| 0.80 | 70.3% | 97.1% | 29.7% |
| 0.90 | 50.3% | 98.8% | 49.7% |
| 0.95 | 23.5% | 99.4% | 76.5% |

- **95% precision** is reached at threshold 0.69, coding **80.3%** of TIRs automatically (actual precision 95.1%).
- **90% precision** is reached at threshold 0.43, coding **96.6%** of TIRs automatically (actual precision 90.1%).

Currently configured `review_threshold`: **0.69**.

### process_sub

| Threshold | Auto-coded | Precision | Left for a coder |
| ---: | ---: | ---: | ---: |
| 0.00 | 100.0% | 80.8% | 0.0% |
| 0.50 | 94.6% | 83.5% | 5.4% |
| 0.60 ← configured | 88.8% | 85.5% | 11.2% |
| 0.70 | 81.7% | 87.7% | 18.3% |
| 0.80 | 72.9% | 89.8% | 27.1% |
| 0.90 | 55.7% | 91.8% | 44.3% |
| 0.95 | 38.7% | 92.4% | 61.3% |

- 95% precision is not reachable at any threshold.
- **90% precision** is reached at threshold 0.82, coding **70.6%** of TIRs automatically (actual precision 90.1%).

Currently configured `review_threshold`: **0.60**.

### process_l3

| Threshold | Auto-coded | Precision | Left for a coder |
| ---: | ---: | ---: | ---: |
| 0.00 | 100.0% | 73.6% | 0.0% |
| 0.50 ← configured | 87.7% | 79.8% | 12.3% |
| 0.60 | 78.8% | 83.4% | 21.2% |
| 0.70 | 69.9% | 86.3% | 30.1% |
| 0.80 | 58.2% | 88.8% | 41.8% |
| 0.90 | 34.1% | 92.3% | 65.9% |
| 0.95 | 17.6% | 93.9% | 82.4% |

- 95% precision is not reachable at any threshold.
- **90% precision** is reached at threshold 0.84, coding **50.8%** of TIRs automatically (actual precision 90.4%).

Currently configured `review_threshold`: **0.50**.

## Which models were kept

Each field's classifiers were gated on validation macro-F1 against a calibrated linear SVM: the other families are carried only where they beat it.

| Field | Kept | Validation macro-F1 |
| --- | --- | --- |
| metric_cat | svm | svm 0.7301 |
| process_cat | svm | svm 0.7109 |
| process_sub | per-parent SVM | see per-parent models |
| process_l3 | per-parent SVM | see per-parent models |
