#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.report_benchmark
# Measures what the trained models are worth in coder time: how much of a file
# can be coded automatically, and at what precision.  Writes
# reports/benchmark.md.
#
#     python -m src.report_benchmark

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:  # modules under src/
    from src.inference import CONFIG, TARGETS, classify, load_bundle, review_threshold
    from src.paths import DATA_DIR, MODEL_DIR, REPORT_DIR
except ImportError:  # flat layout, modules at the repository root
    from inference import CONFIG, TARGETS, classify, load_bundle, review_threshold
    from paths import DATA_DIR, MODEL_DIR, REPORT_DIR


PROC_DIR = DATA_DIR / "processed"

# The precision levels a coding team would plausibly want to hold the
# automatic answers to.
PRECISION_TARGETS = [0.95, 0.90]
SWEEP = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


# Finds the lowest threshold that still meets a precision target.
def threshold_for(confidence: np.ndarray, correct: np.ndarray, target: float):
    """Return (threshold, coverage, precision) or None if unreachable.

    Scans upward and takes the first threshold that holds, which is the one
    that leaves the most rows automatically coded.
    """
    for threshold in np.arange(0.05, 1.0, 0.01):
        keep = confidence >= threshold
        if keep.sum() < 50:
            continue
        precision = float(correct[keep].mean())
        if precision >= target:
            return float(threshold), float(keep.mean()), precision
    return None


# Builds the report.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=str(PROC_DIR / "test.csv"))
    parser.add_argument("--out", default=str(REPORT_DIR / "benchmark.md"))
    args = parser.parse_args()

    df = pd.read_csv(args.split, low_memory=False)
    bundle = load_bundle()
    preds = classify(df["text"].astype(str).tolist(), bundle)

    lines: List[str] = [
        "# TIR coding benchmark",
        "",
        "What the models are worth in coder time. Regenerate with "
        "`python -m src.report_benchmark`.",
        "",
        f"Held-out split: **{len(df):,} rows**, never seen during training.",
        "",
        "Accuracy alone is a poor guide here — the categories are very unbalanced, so a "
        "model can score well while getting the whole rare tail wrong. Macro-F1 and the "
        "coverage curve below are the numbers that matter.",
        "",
        "## Accuracy by field",
        "",
        "| Field | Rows judged | Accuracy | Macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]

    summary = {}
    for target in bundle["order"]:
        col = f"label_{target}"
        if col not in df.columns:
            continue
        id2name = bundle["targets"][target]["id2name"]
        truth = df[col].map(lambda i: id2name.get(int(i), "") if int(i) >= 0 else "")
        judged = (truth != "") & (preds[f"pred_{target}"] != "")
        if not judged.any():
            continue

        y_true = truth[judged]
        y_pred = preds.loc[judged, f"pred_{target}"]
        accuracy = accuracy_score(y_true, y_pred)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        summary[target] = {
            "judged": judged,
            "correct": (y_true.to_numpy() == y_pred.to_numpy()),
            "confidence": preds.loc[judged, f"confidence_{target}"].to_numpy(),
        }
        lines.append(
            f"| {target} | {int(judged.sum()):,} | {accuracy:.4f} | {macro:.4f} |"
        )

    lines += [
        "",
        "> Read each figure against how consistently people code that field — see "
        "`data_sufficiency.md`. A model matching its coders is at the ceiling, not failing.",
        "",
        "## Coverage at a precision target",
        "",
        "The number that converts to time saved: at a given confidence threshold, what share "
        "of TIRs can be coded automatically, and how often is that automatic answer right?",
        "",
    ]

    for target, entry in summary.items():
        confidence, correct = entry["confidence"], entry["correct"]
        lines += [
            f"### {target}",
            "",
            "| Threshold | Auto-coded | Precision | Left for a coder |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for threshold in SWEEP:
            keep = confidence >= threshold
            if keep.sum() == 0:
                continue
            marker = " ← configured" if abs(threshold - review_threshold(target)) < 0.005 else ""
            lines.append(
                f"| {threshold:.2f}{marker} | {keep.mean():.1%} | "
                f"{correct[keep].mean():.1%} | {(~keep).mean():.1%} |"
            )

        lines.append("")
        for precision in PRECISION_TARGETS:
            found = threshold_for(confidence, correct, precision)
            if found:
                threshold, coverage, achieved = found
                lines.append(
                    f"- **{precision:.0%} precision** is reached at threshold "
                    f"{threshold:.2f}, coding **{coverage:.1%}** of TIRs automatically "
                    f"(actual precision {achieved:.1%})."
                )
            else:
                lines.append(f"- {precision:.0%} precision is not reachable at any threshold.")
        lines += [
            "",
            f"Currently configured `review_threshold`: **{review_threshold(target):.2f}**.",
            "",
        ]

    lines += [
        "## Which models were kept",
        "",
        "Each field's classifiers were gated on validation macro-F1 against a calibrated "
        "linear SVM: the other families are carried only where they beat it.",
        "",
        "| Field | Kept | Validation macro-F1 |",
        "| --- | --- | --- |",
    ]
    for target in bundle["order"]:
        spec_path = MODEL_DIR / target / "ensemble.json"
        if not spec_path.is_file():
            routing = MODEL_DIR / target / "routing.json"
            if routing.is_file():
                lines.append(f"| {target} | per-parent SVM | see per-parent models |")
            continue
        spec = json.loads(spec_path.read_text())
        scored = spec.get("validation_macro_f1", {})
        lines.append(
            f"| {target} | {', '.join(spec.get('weights', {}))} | "
            + ", ".join(f"{k} {v:.4f}" for k, v in scored.items())
            + " |"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"✔ Wrote {out_path}")


if __name__ == "__main__":
    main()
