#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.classify
# Classifies a raw QPS export from the command line and reports accuracy
# wherever the file carries the true categories.

import argparse
import json
from pathlib import Path
from typing import cast
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import (
    CONFIG, DATA_DIR, MODEL_DIR, REQUIRED_COLS, TARGETS, TEXT_COLS, target_title,
)
from src.data import (
    build_text_series, canonicalize, normalize_categories, validate_input_dataframe,
)
from src.inference import classify, load_bundle


# Reports how well one target did against the file's own labels.
def report_target(df: pd.DataFrame, target: str, spec: dict) -> None:
    """Print accuracy and macro-F1 for `target` where truth is available.

    Macro-F1 is reported alongside accuracy because the categories are very
    unbalanced: a model can score in the high eighties while getting every
    rare category wrong, and only the macro average shows it.
    """
    column = spec["column"]
    if column not in df.columns:
        return

    table = CONFIG.get(spec.get("normalization", ""), {})
    truth = normalize_categories(cast(pd.Series, df[column]), table, "")
    predicted = df[f"pred_{target}"]

    known = truth.notna() & (truth.astype(str).str.strip() != "") & (predicted != "")
    if not known.any():
        print(f"\n⚠ {target}: no rows carry a known '{column}' value; metrics skipped.")
        return

    y_true, y_pred = truth[known], predicted[known]
    print(f"\n=== {target} ({target_title(target)}) — {int(known.sum()):,} labelled rows ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)  # type: ignore[arg-type]
    print(f"Macro-F1 : {macro:.4f}\n")
    # zero_division=0 reports a category the models never predicted as 0.0
    # rather than warning.  sklearn documents 0 as valid but annotates the
    # parameter as str, hence the ignore.
    print(classification_report(
        y_true, y_pred, digits=4,
        zero_division=0,  # type: ignore[arg-type]
    ))


# Runs inference on a CSV or Excel input file.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True, help="CSV or Excel file to classify")
    parser.add_argument("--out", required=True, help="Where to write the predictions CSV")
    parser.add_argument("--models_dir", default=str(MODEL_DIR))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Print the report without writing the predictions file",
    )
    args = parser.parse_args()

    fp = Path(args.raw_csv)
    raw = pd.read_csv(fp) if fp.suffix.lower() == ".csv" else pd.read_excel(fp)
    df = canonicalize(raw)

    validate_input_dataframe(df, REQUIRED_COLS)
    df["text"] = build_text_series(df, TEXT_COLS)

    bundle = load_bundle(Path(args.models_dir), Path(args.data_dir))
    preds = classify(df["text"].tolist(), bundle)
    out = pd.concat([df.reset_index(drop=True), preds], axis=1)

    for target, spec in TARGETS.items():
        report_target(out, target, spec)

    if not args.metrics_only:
        Path(args.out).write_text(out.to_csv(index=False))
        print("\n✔ Saved predictions:", args.out)


if __name__ == "__main__":
    main()
