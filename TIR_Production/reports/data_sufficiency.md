# TIR data sufficiency

Is the QPS data labelled well enough to train a classifier on? Regenerate with `python -m src.report_data --raw_csv <files>`.

## Files read

| File | Rows | Fields recognised | Not present |
| --- | ---: | ---: | --- |
| QPS Pull Jan 2023 - Dec 2025 Non Nuc_Norforn-N_TIR_SURV_Validated.xlsx | 94,505 | 15/16 | item_id |
| TIR Export Example1.xlsx | 8,776 | 16/16 | — |

Combined: **103,281 rows**, of which **12,321** were duplicates across or within files, leaving **90,960**.

The smaller export is very largely contained in the larger one. Because the two name their columns differently, a whole-row comparison finds nothing in common; de-duplication is done on the canonical text and labels instead.

## Label coverage

| Field | Rows coded | Coverage | Categories | Classes under the minimum |
| --- | ---: | ---: | ---: | ---: |
| metric_cat | 90,958 | 100.0% | 7 | 0 (under 3) |
| process_cat | 61,263 | 67.4% | 27 | 0 (under 3) |
| process_sub | 60,807 | 66.9% | 144 | 35 (under 10) |
| process_l3 | 56,927 | 62.6% | 543 | 537 (under 10) |

## Coder consistency — the accuracy ceiling

Where the same Description 1 was coded more than once, did it get the same code? A model cannot be more consistent than the data it learns from, so these figures are the ceiling every accuracy number should be read against.

| Field | Repeated descriptions | Rows | Given conflicting codes | Agreement with majority |
| --- | ---: | ---: | ---: | ---: |
| metric_cat | 1,974 | 25,855 | 632 (32.0%) | **91.4%** |
| process_cat | 932 | 8,397 | 291 (31.2%) | **85.3%** |
| process_sub | 897 | 7,978 | 389 (43.4%) | **75.4%** |
| process_l3 | 716 | 5,758 | 374 (52.2%) | **67.0%** |

> These are a **lower bound** on agreement. Two TIRs can share a Description 1 and still be different events, so some of what is counted here as disagreement is two coders correctly coding two different things. Confirming a sample of the conflicting pairs with the coding team would firm this up before the figures are quoted.

## Verdict

| Field | Is the data sufficient to train on? |
| --- | --- |
| metric_cat | **sufficient** |
| process_cat | **sufficient** |
| process_sub | **marginal** |
| process_l3 | **insufficient for a single confident answer** (and 537 of 1080 classes have too few records to learn) |

## Why this uses TF-IDF and a linear model, not a language model

Small and large language models are both listed as areas of research for this project, and this system uses neither: the text is encoded with word and character TF-IDF and classified with a calibrated linear SVM.

That is a deliberate constraint rather than an oversight. The dependency list is explicit that the pipeline needs numpy only — no torch, no downloaded model file — which keeps it deterministic, auditable and installable offline, all of which matter more here than the last point of accuracy.

What a language model would plausibly add is the rare-category tail, where there are too few examples for a bag-of-features model to generalise. What it would cost is model provenance and approval to run in this environment. It would **not** lift the ceiling above: that is set by how consistently the training labels were assigned, and no model can be more consistent than its data.
