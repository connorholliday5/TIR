#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.report_data
# Answers the question the study actually asks: is the QPS data labelled well
# enough to train on?  Writes reports/data_sufficiency.md.
#
#     python -m src.report_data --raw_csv "<export>.xlsx" ...

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:  # modules under src/
    from src.columns import canonicalize, resolve_columns
    from src.paths import CONFIG_PATH, REPORT_DIR, ROOT
    from src.utils import build_text_series
except ImportError:  # flat layout, modules at the repository root
    from columns import canonicalize, resolve_columns
    from paths import CONFIG_PATH, REPORT_DIR, ROOT
    from utils import build_text_series


CONFIG = json.loads(CONFIG_PATH.read_text())
ALIASES: dict = CONFIG.get("column_aliases", {})
TARGETS: dict = CONFIG.get("targets", {})
TEXT_COLS = CONFIG.get("text_columns", ["description_1"])

# A target is called sufficient when coders agree with each other at least
# this often on identical text; marginal down to the second figure; below it,
# a single confident answer is not something the data can support.
SUFFICIENT = 0.85
MARGINAL = 0.70


# Measures how consistently people coded the same text.
def consistency(df: pd.DataFrame, column: str) -> dict:
    """Return agreement statistics for repeated descriptions.

    Where the same description appears more than once, the codes it was given
    are compared.  Disagreement puts a ceiling on any model: it cannot learn a
    rule that the training data itself does not follow consistently.

    Repeats are found on Description 1 alone rather than on the full model
    input.  Adding the title and second description makes near-identical TIRs
    look distinct and shrinks the comparable set roughly tenfold, leaving too
    few groups per field to say anything steady; Description 1 is the field a
    coder reads first and gives a sample an order of magnitude larger.

    This is a lower bound on real agreement.  Two TIRs can share a Description 1
    and still be different events — a different drawing, compartment or
    Description 2 — so some of what is counted here as disagreement is two
    coders correctly coding two different things.
    """
    usable = df[df[column].notna() & (df[column].astype(str).str.strip() != "")]
    if usable.empty:
        return {"groups": 0, "rows": 0, "conflicting": 0, "agreement": float("nan")}

    grouped = usable.groupby("repeat_key")[column]
    sizes = grouped.size()
    repeated = sizes[sizes > 1].index
    if len(repeated) == 0:
        return {"groups": 0, "rows": 0, "conflicting": 0, "agreement": float("nan")}

    subset = usable[usable["repeat_key"].isin(repeated)]
    per_text = subset.groupby("repeat_key")[column]
    conflicting = int((per_text.nunique() > 1).sum())

    majority = per_text.agg(lambda s: s.mode().iloc[0])
    agreement = float((subset[column].values == subset["repeat_key"].map(majority).values).mean())

    return {
        "groups": int(len(repeated)),
        "rows": int(len(subset)),
        "conflicting": conflicting,
        "agreement": agreement,
    }


# Classifies a target as sufficient, marginal or insufficient.
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


# Builds the report.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", nargs="+", required=True)
    parser.add_argument("--out", default=str(REPORT_DIR / "data_sufficiency.md"))
    args = parser.parse_args()

    lines: List[str] = [
        "# TIR data sufficiency",
        "",
        "Is the QPS data labelled well enough to train a classifier on? "
        "Regenerate with `python -m src.report_data --raw_csv <files>`.",
        "",
        "## Files read",
        "",
        "| File | Rows | Fields recognised | Not present |",
        "| --- | ---: | ---: | --- |",
    ]

    frames = []
    for name in args.raw_csv:
        path = Path(name)
        if not path.exists():
            path = ROOT / Path(name).name
        raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        resolved = resolve_columns(raw, ALIASES)
        absent = [n for n in ALIASES if n not in resolved] or ["—"]
        lines.append(
            f"| {path.name} | {len(raw):,} | {len(resolved)}/{len(ALIASES)} | {', '.join(absent)} |"
        )
        frames.append(canonicalize(raw, ALIASES))

    merged = pd.concat(frames, ignore_index=True)
    merged["text"] = build_text_series(merged, TEXT_COLS)

    # Repeated-description comparisons key on Description 1 by itself; see
    # `consistency` for why the full model input is the wrong grain here.
    merged["repeat_key"] = build_text_series(merged, ["description_1"])

    label_columns = [spec["column"] for spec in TARGETS.values() if spec["column"] in merged.columns]
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text", *label_columns], keep="first")

    lines += [
        "",
        f"Combined: **{before:,} rows**, of which **{before - len(merged):,}** were duplicates "
        f"across or within files, leaving **{len(merged):,}**.",
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
        values = merged[column]
        filled = values.notna() & (values.astype(str).str.strip() != "")
        counts = values[filled].value_counts()
        minimum = int(spec.get("min_class_size", 3))
        thin = int((counts < minimum).sum())
        stats[target] = {
            "classes": int((counts >= minimum).sum()),
            "thin": thin,
            "coverage": float(filled.mean()),
        }
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
        share = (
            f"{result['conflicting'] / result['groups']:.1%}" if result["groups"] else "—"
        )
        agreement = (
            f"**{result['agreement']:.1%}**" if result["agreement"] == result["agreement"] else "—"
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
    for target in TARGETS:
        if target not in stats:
            continue
        entry = stats[target]
        lines.append(
            f"| {target} | {verdict(entry.get('agreement', float('nan')), entry.get('classes', 0), entry.get('thin', 0))} |"
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
        "",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"✔ Wrote {out_path}")


if __name__ == "__main__":
    main()
