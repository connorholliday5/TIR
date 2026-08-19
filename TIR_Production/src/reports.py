#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.reports
# The two questions this study asks, answered as regenerable reports:
#
#   sufficiency — is the QPS data labelled well enough to train on?
#   benchmark   — what are the trained models worth in coder time?
#
#     python -m src.reports sufficiency --raw_csv "<export>.xlsx" ...
#     python -m src.reports benchmark
#     python -m src.reports all --raw_csv "<export>.xlsx" ...

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, cast
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    ALIASES, MODEL_DIR, PROC_DIR, REPORT_DIR, ROOT, TARGETS, TEXT_COLS,
    review_threshold,
)
from src.data import build_text_series, canonicalize, resolve_columns
from src.inference import classify, load_bundle


# A field is called sufficient when coders agree with each other at least this
# often on identical text; marginal down to the second figure; below it, a
# single confident answer is not something the data can support.
SUFFICIENT = 0.85
MARGINAL = 0.70

# The precision levels a coding team would plausibly hold automatic answers
# to, and the thresholds the coverage curve is sampled at.
PRECISION_TARGETS = [0.95, 0.90]
SWEEP = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


# Writes a report, creating its directory.
def _write(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"✔ Wrote {path}")


# -- sufficiency -------------------------------------------------------------


# Measures how consistently people coded the same text.
def consistency(df: pd.DataFrame, column: str) -> dict:
    """Return agreement statistics for repeated descriptions.

    Where the same description appears more than once, the codes it was given
    are compared.  Disagreement puts a ceiling on any model: it cannot learn a
    rule the training data itself does not follow.

    Repeats are found on Description 1 alone rather than on the full model
    input.  Adding the title and second description makes near-identical TIRs
    look distinct and shrinks the comparable set roughly tenfold, leaving too
    few groups per field to say anything steady.

    This is a lower bound on real agreement: two TIRs can share a Description 1
    and still be different events — a different drawing, compartment or
    Description 2 — so some of what is counted here as disagreement is two
    coders correctly coding two different things.
    """
    usable = cast(pd.DataFrame, df[df[column].notna() & (df[column].astype(str).str.strip() != "")])
    empty = {"groups": 0, "rows": 0, "conflicting": 0, "agreement": float("nan")}
    if usable.empty:
        return empty

    # pandas-stubs types size() and value_counts() as ndarray; both return a
    # Series here, and the code below relies on its index.
    sizes = cast(pd.Series, usable.groupby("repeat_key")[column].size())
    repeated = cast(pd.Series, sizes[sizes > 1]).index
    if len(repeated) == 0:
        return empty

    keys = cast(pd.Series, usable["repeat_key"])
    subset = cast(pd.DataFrame, usable[keys.isin(pd.Series(repeated))])
    per_text = subset.groupby("repeat_key")[column]

    # transform broadcasts each group's most common code back onto its own
    # rows, so the comparison below is row-against-its-own-majority without
    # having to build and re-join a lookup.
    majority_per_row = per_text.transform(lambda s: s.mode().iloc[0])

    return {
        "groups": int(len(repeated)),
        "rows": int(len(subset)),
        "conflicting": int((per_text.nunique() > 1).sum()),
        "agreement": float(
            np.mean(np.asarray(subset[column]) == np.asarray(majority_per_row))
        ),
    }


# Classifies a field as sufficient, marginal or insufficient.
def verdict(agreement: float, classes: int, thin: int) -> str:
    if agreement != agreement:  # NaN
        return "unknown — no repeated descriptions to compare"
    if agreement >= SUFFICIENT:
        base = "**sufficient**"
    elif agreement >= MARGINAL:
        base = "**marginal**"
    else:
        base = "**insufficient for a single confident answer**"
    if thin and thin > classes / 2:
        base += f" (and {thin} of {classes + thin} classes have too few records to learn)"
    return base


# Builds the data-sufficiency report.
def build_sufficiency(raw_csv: List[str], out: Path) -> None:
    """Answer whether the supplied exports are labelled well enough to train on."""
    lines: List[str] = [
        "# TIR data sufficiency",
        "",
        "Is the QPS data labelled well enough to train a classifier on? "
        "Regenerate with `python -m src.reports sufficiency --raw_csv <files>`.",
        "",
        "## Files read",
        "",
        "| File | Rows | Fields recognised | Not present |",
        "| --- | ---: | ---: | --- |",
    ]

    frames = []
    for name in raw_csv:
        path = Path(name)
        if not path.exists():
            path = ROOT / Path(name).name
        if not path.exists():
            path = ROOT.parent / Path(name).name
        raw = (
            pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"}
            else pd.read_csv(path)
        )
        resolved = resolve_columns(raw)
        missing = [name for name in ALIASES if name not in resolved] or ["—"]
        lines.append(
            f"| {path.name} | {len(raw):,} | {len(resolved)}/{len(ALIASES)} | "
            f"{', '.join(missing)} |"
        )
        frames.append(canonicalize(raw))

    merged = pd.concat(frames, ignore_index=True)
    merged["text"] = build_text_series(merged, TEXT_COLS)
    # Repeated-description comparisons key on Description 1 by itself; see
    # `consistency` for why the full model input is the wrong grain.
    merged["repeat_key"] = build_text_series(merged, ["description_1"])

    label_columns = [
        spec["column"] for spec in TARGETS.values() if spec["column"] in merged.columns
    ]
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text", *label_columns], keep="first")

    lines += [
        "",
        f"Combined: **{before:,} rows**, of which **{before - len(merged):,}** were "
        f"duplicates across or within files, leaving **{len(merged):,}**.",
        "",
        "The smaller export is very largely contained in the larger one. Because the two "
        "name their columns differently, a whole-row comparison finds nothing in common; "
        "de-duplication is done on the canonical text and labels instead.",
        "",
        "## Label coverage",
        "",
        "| Field | Rows coded | Coverage | Categories | Classes under the minimum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    stats: Dict[str, dict] = {}
    for target, spec in TARGETS.items():
        column = spec["column"]
        if column not in merged.columns:
            continue
        filled = merged[column].notna() & (merged[column].astype(str).str.strip() != "")
        counts = merged.loc[filled, column].value_counts()
        minimum = int(spec.get("min_class_size", 3))
        thin = int((counts < minimum).sum())
        stats[target] = {"classes": int((counts >= minimum).sum()), "thin": thin}
        lines.append(
            f"| {target} | {int(filled.sum()):,} | {filled.mean():.1%} | "
            f"{stats[target]['classes']} | {thin} (under {minimum}) |"
        )

    lines += [
        "",
        "## Coder consistency — the accuracy ceiling",
        "",
        "Where the same Description 1 was coded more than once, did it get the same code? "
        "A model cannot be more consistent than the data it learns from, so these figures "
        "are the ceiling every accuracy number should be read against.",
        "",
        "| Field | Repeated descriptions | Rows | Given conflicting codes | Agreement with majority |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for target, spec in TARGETS.items():
        column = spec["column"]
        if column not in merged.columns:
            continue
        result = consistency(merged, column)
        stats.setdefault(target, {})["agreement"] = result["agreement"]
        share = f"{result['conflicting'] / result['groups']:.1%}" if result["groups"] else "—"
        agreement = (
            f"**{result['agreement']:.1%}**"
            if result["agreement"] == result["agreement"] else "—"
        )
        lines.append(
            f"| {target} | {result['groups']:,} | {result['rows']:,} | "
            f"{result['conflicting']:,} ({share}) | {agreement} |"
        )

    lines += [
        "",
        "> These are a **lower bound** on agreement. Two TIRs can share a Description 1 and "
        "still be different events, so some of what is counted here as disagreement is two "
        "coders correctly coding two different things. Confirming a sample of the conflicting "
        "pairs with the coding team would firm this up before the figures are quoted.",
        "",
        "## Verdict",
        "",
        "| Field | Is the data sufficient to train on? |",
        "| --- | --- |",
    ]
    for target, entry in stats.items():
        lines.append(
            f"| {target} | "
            f"{verdict(entry.get('agreement', float('nan')), entry.get('classes', 0), entry.get('thin', 0))} |"
        )

    lines += [
        "",
        "## Why this uses TF-IDF and a linear model, not a language model",
        "",
        "Small and large language models are both listed as areas of research for this "
        "project, and this system uses neither: the text is encoded with word and character "
        "TF-IDF and classified with a calibrated linear SVM.",
        "",
        "That is a deliberate constraint rather than an oversight. The dependency list is "
        "explicit that the pipeline needs numpy only — no torch, no downloaded model file — "
        "which keeps it deterministic, auditable and installable offline, all of which "
        "matter more here than the last point of accuracy.",
        "",
        "What a language model would plausibly add is the rare-category tail, where there "
        "are too few examples for a bag-of-features model to generalise. What it would cost "
        "is model provenance and approval to run in this environment. It would **not** lift "
        "the ceiling above: that is set by how consistently the training labels were "
        "assigned, and no model can be more consistent than its data.",
    ]

    _write(out, lines)


# -- benchmark ---------------------------------------------------------------


# Resolves a stored label id back to its category name.
def label_to_name(value: Any, id2name: Dict[int, str]) -> str:
    """Return the category `value` refers to, or "" when it refers to none.

    Preprocessing writes -1 for "this row carries no code for this field", but
    the label columns round-trip through CSV, so a row can also arrive as NA —
    and int(NA) raises ValueError rather than returning anything, which would
    take down the whole report over one malformed row.
    """
    if pd.isna(value):
        return ""
    label = int(value)
    return id2name.get(label, "") if label >= 0 else ""


# Finds the lowest threshold that still meets a precision target.
def threshold_for(confidence: np.ndarray, correct: np.ndarray, target: float):
    """Return (threshold, coverage, precision), or None if unreachable.

    Scans upward and takes the first threshold that holds, which is the one
    leaving the most rows coded automatically.
    """
    for threshold in np.arange(0.05, 1.0, 0.01):
        keep = confidence >= threshold
        if keep.sum() < 50:
            continue
        precision = float(correct[keep].mean())
        if precision >= target:
            return float(threshold), float(keep.mean()), precision
    return None


# Builds the benchmark report.
def build_benchmark(split: str, out: Path) -> None:
    """Measure what the trained models are worth in coder time."""
    df = pd.read_csv(split, low_memory=False)
    bundle = load_bundle()
    preds = classify(df["text"].astype(str).tolist(), bundle)

    lines: List[str] = [
        "# TIR coding benchmark",
        "",
        "What the models are worth in coder time. Regenerate with "
        "`python -m src.reports benchmark`.",
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
        truth = df[col].map(lambda v: label_to_name(v, id2name))
        judged = (truth != "") & (preds[f"pred_{target}"] != "")
        if not judged.any():
            continue

        y_true = cast(pd.Series, truth[judged])
        y_pred = cast(pd.Series, preds.loc[judged, f"pred_{target}"])
        summary[target] = {
            "correct": (y_true.to_numpy() == y_pred.to_numpy()),
            "confidence": preds.loc[judged, f"confidence_{target}"].to_numpy(),
        }
        lines.append(
            f"| {target} | {int(judged.sum()):,} | {accuracy_score(y_true, y_pred):.4f} | "
            # zero_division=0 reports a category the models never predicted as
            # 0.0 rather than warning.  sklearn documents 0 as valid but
            # annotates the parameter as str, hence the ignore.
            f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f} |"  # type: ignore[arg-type]
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
        lines += ["", f"Currently configured `review_threshold`: **{review_threshold(target):.2f}**.", ""]

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
        if spec_path.is_file():
            spec = json.loads(spec_path.read_text())
            scored = spec.get("validation_macro_f1", {})
            lines.append(
                f"| {target} | {', '.join(spec.get('weights', {}))} | "
                + ", ".join(f"{k} {v:.4f}" for k, v in scored.items()) + " |"
            )
        elif (MODEL_DIR / target / "routing.json").is_file():
            lines.append(f"| {target} | per-parent SVM | see per-parent models |")

    _write(out, lines)


# Runs whichever report was asked for.
def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the study's reports.")
    parser.add_argument("report", choices=["sufficiency", "benchmark", "all"])
    parser.add_argument("--raw_csv", nargs="+", help="Exports to read (sufficiency)")
    parser.add_argument("--split", default=str(PROC_DIR / "test.csv"))
    parser.add_argument("--out-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.report in ("sufficiency", "all"):
        if not args.raw_csv:
            parser.error("sufficiency needs --raw_csv naming the exports to read")
        build_sufficiency(args.raw_csv, out_dir / "data_sufficiency.md")

    if args.report in ("benchmark", "all"):
        build_benchmark(args.split, out_dir / "benchmark.md")


if __name__ == "__main__":
    main()
