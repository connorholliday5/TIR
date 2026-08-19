✅ Improvements You Should Make (Prioritized)
HIGH PRIORITY (Correctness & Risk)


Reduce complexity in inference.classify and _predict_child

These two functions carry the highest defect risk.
Break into smaller helpers: parent retrieval, child routing, blank handling, thresholding, override handling.



Add model challengers for deep fields (process_sub, process_l3)

train.py calls fit_svm directly → deep fields are always SVM-only.
Add optional LR/SGD/XGB gating for deeper levels.



Revisit equal-weight blending for ensemble models

Currently, if SVM+SGD survive gate, they get 50/50 even if SGD beat SVM by several points.
Add a weighted scheme:

weight ∝ validation macro-F1
or allow manual override in config.





Add specific tests for hierarchical routing

Verify correct parent mapping
Verify alternates (top_k) returned correctly
Verify reviewer override cascades as intended



Strengthen NA / None handling in several modules

Two NA bugs were found; assume similar defects remain.
Add explicit NA guards in inference/classify, feedback, preprocess.



Improve normalization error logging

When a category is unmapped, warn with structured logging.
Current print statements are good but could be easier to aggregate.



MEDIUM PRIORITY (Maintainability / Design)


Add module-level author/revision headers (CST‑3)

Your team should decide the convention.
Apply once agreed.



Decide future of api.py

If unused → remove and simplify dependency footprint.
If kept → add tests and documentation.



Decompose long “main” functions

train.main, preprocess.main, reports.build_sufficiency, etc.
These exceed complexity thresholds primarily due to orchestration.



Add tests for alternates (top‑k outputs)


Convert classify.py’s CSV write from write_text() → to_csv()



Prevent CSV embedding inside a text file
Avoid memory spike on huge outputs.

LOW PRIORITY (Tooling / Nice-to-have)

Add Streamlit UI regression tests
Add Excel workbook formatting tests
Document retirement of REQ‑013 in SDD
Add richer logging around reviewer overrides
Add config-driven ability to:

tune top_k
set per-target feature limits
optionally route by prefix rules for hierarchy analysis




✅ Answers (or Best-Path Guidance) to the Section 8 Reviewer Questions
1. Why is coder agreement grouped on Description 1 rather than full model input?
Answer:
Because grouping by the full model input shrinks the number of comparable repeated items by ~90%.

Description 2 is missing on ~72% of TIRs
Doc Title varies widely between exports
Two TIRs can share Description 1 and legitimately differ in subtle ways.
Therefore, Description 1 gives a lower-bound estimate of agreement without removing 90% of usable pairs.

2. Why does the hierarchy come from observed parent→child pairs rather than code prefixes?
Answer:
Because code prefixes disagree with actual practice in about:

0.1% of Process Sub
4.5% of Level 3
Prefix-based nesting is a rule the taxonomy should follow, but observed coder behavior does not always match it.
If prefix rules were enforced:
some real pairs would be excluded
some impossible pairs would be included
Observed hierarchy ensures the model only predicts combinations that really occur in historical data.

3. What happens on an export whose layout no alias covers — loud or quiet?
Answer:
It fails loudly at two points:

resolve_columns() → missing aliases
validate_input_dataframe() → missing required columns
This produces explicit errors:
“missing required column(s) …”
“absent from this file: …”
The failure mode is intentional — quiet acceptance would corrupt the dataset.

4. Boosting scored worst on every field. Why — the method or the feature limit?
Answer:
It is almost entirely the feature limit, not the method.
XGBoost:

cannot handle 183,000 TF‑IDF features
kernel killed full-width runs twice
reducing to 30,000 features picks top χ² features, favoring frequent categories
Deep-level categories are too rare → trees cannot learn meaningful splits.

If full-width boosting could run:

It might compete with SVM in top-level fields
But would still likely struggle on deep taxonomy tail

5. Is inference.classify worth decomposing, or genuinely irreducible?
Answer:
It is worth decomposing.
Reasons:

complexity = 19 (highest in project)
contains multiple responsibilities:

feature construction
blank handling
override logic
parent routing
child prediction
alternate return
review threshold inheritance



Separating it into ~4 helper functions:

reduces risk
improves readability
makes targeted tests easier
lets you isolate hierarchical routing (biggest risk area)
