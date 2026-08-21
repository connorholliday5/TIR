#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.data
# Implements:
#   - REQ-002 (Text cleaning)
#   - REQ-011 (Input validation)
# Reading a raw QPS export: recognising its columns whatever they are called,
# and turning them into the single text field the models are trained on.

import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, cast
import pandas as pd

from src.config import ALIASES


# -- column recognition ------------------------------------------------------
#
# The TIR team exports the same fields under different headings depending on
# which report produced the file: "Description 1", "DESCRIPTION_ONE" and "Item
# Description 1" are all the same field.  Every other module reads canonical
# names, and this is the only place that knows what a given workbook calls
# them.  The alias table lives in config.json, so a fourth export layout is a
# configuration change rather than a code change.


# Reduces a heading to the form used for matching.
def _key(name: str) -> str:
    """Fold a column heading to its comparison key.

    Case, surrounding whitespace, underscores and repeated spaces all vary
    between exports of the same field ("DESCRIPTION_ONE" against "Description
    One"), and none of them carry meaning, so they are normalised away rather
    than enumerated as separate aliases.
    """
    return re.sub(r"[\s_]+", " ", str(name).strip().lower()).strip()


# Finds the actual column backing each canonical name.
def resolve_columns(
    df: pd.DataFrame, aliases: Optional[Mapping[str, Sequence[str]]] = None
) -> Dict[str, str]:
    """Return {canonical name: the column of `df` that holds it}.

    Canonical names the frame does not carry are simply absent from the
    result, so callers can distinguish "this export has no Description 2" from
    "Description 2 is empty".
    """
    aliases = ALIASES if aliases is None else aliases

    available: Dict[str, str] = {}
    for column in df.columns:
        # First occurrence wins: a workbook with two columns folding to the
        # same key keeps the leftmost, which is the one a reader sees first.
        available.setdefault(_key(column), str(column))

    resolved: Dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            match = available.get(_key(name))
            if match is not None:
                resolved[canonical] = match
                break

    return resolved


# Renames a raw export's columns to the canonical names.
def canonicalize(
    df: pd.DataFrame, aliases: Optional[Mapping[str, Sequence[str]]] = None
) -> pd.DataFrame:
    """Return a copy of `df` with recognised columns renamed to canonical form.

    Columns no alias claims are left exactly as they were, so an uploaded file
    keeps every field it arrived with and the batch tab can hand them back.
    """
    aliases = ALIASES if aliases is None else aliases
    resolved = resolve_columns(df, aliases)

    rename = {}
    for canonical, actual in resolved.items():
        if actual == canonical:
            continue
        if canonical in df.columns:
            # The frame already has a column by this name and a different one
            # claims to be it; keeping both would make the result depend on
            # column order, so the existing one stands.
            continue
        rename[actual] = canonical

    return df.rename(columns=rename)


# Names the canonical fields an export is missing.
def missing_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    aliases: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[str]:
    """Return the required canonical names that `df` cannot supply."""
    aliases = ALIASES if aliases is None else aliases
    resolved = resolve_columns(df, aliases)
    return [n for n in required if n not in resolved and n not in df.columns]


# Checks that every column the pipeline needs is present.
def validate_input_dataframe(df: pd.DataFrame, required: List[str]) -> None:
    """Raise a ValueError if any of the required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required column(s): {', '.join(missing)}"
        )


# Reads one raw export with its columns already canonicalised.
def load_export(path: Path) -> pd.DataFrame:
    """Load an Excel or CSV export, renamed to canonical column names."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    return canonicalize(df)


# -- text -------------------------------------------------------------------


# Standardises raw description text.
def clean_text(text: str) -> str:
    """Standardise raw description text.

    Collapses every run of whitespace — spaces, tabs, new-lines — to a single
    space, then trims and lower-cases.  Character n-grams see a doubled space,
    and 2% of the sample export contains one, so those records would otherwise
    classify differently for a reason that carries no meaning.
    """
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# Reports whether a text has anything left to classify.
def is_blank_text(text) -> bool:
    """True when `text` is empty, missing, or only whitespace.

    A blank description still produces a prediction, because the models score
    an all-zero feature vector rather than refusing, and that prediction can
    look confident: an empty string scored 99.8% for one target.  Callers use
    this to withhold the answer instead of presenting a category nothing was
    read from.
    """
    return not str(text).strip() or str(text).strip().lower() in {"nan", "none"}


# The spellings a blank category arrives as once it has been through astype.
_BLANK_VALUES = ["", "nan", "none", "nat", "<na>"]


# Marks the rows where a category column carries no value.
def blank_values(values: pd.Series) -> pd.Series:
    """Boolean Series, True where `values` holds no category.

    "Nobody coded this" and "coded as something unrecognised" are different
    facts and used to be conflated, which fabricated an OTHER category holding
    a third of the training rows.  Both preprocessing (deciding which rows can
    train a target) and branch labelling (deciding whether a whole family of
    codes applies to a record) depend on drawing that line the same way, so
    they draw it here rather than each writing the check out.

    Tested against `isna` as well as the placeholder spellings: pandas 2
    renders a missing value as the literal string "nan" under `astype(str)`
    while pandas 3 keeps it missing, so checking only one of the two silently
    does nothing on the other version.
    """
    raw = cast(pd.Series, values).astype(str).str.strip()
    return cast(pd.Series, values).isna() | raw.fillna("").str.lower().isin(_BLANK_VALUES)


# Joins the configured description columns into the model's input text.
def build_text_series(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Join `columns` into the single text field the models are trained on.

    Driven by `text_columns` in config.json, so training, batch inference and
    the UI always compose their input the same way.

    Columns the frame does not carry are skipped rather than raised on: the
    three QPS export layouts do not all include a Description 2, and a file
    that omits one should still classify on what it does have.

    `fillna` runs before `astype`: on pandas 2.x `astype(str)` renders a
    missing value as the literal string "nan", which `fillna` can no longer
    replace, appending a phantom token to every row.
    """
    if not columns:
        raise ValueError("At least one description column is required.")

    present = [name for name in columns if name in df.columns]
    if not present:
        raise ValueError(
            f"None of the text column(s) {', '.join(columns)} are in the input; "
            f"it has: {', '.join(str(c) for c in df.columns[:12])}…"
        )

    # Subscripting a DataFrame is typed as returning either a Series or a
    # DataFrame; the column names here always select a single Series.
    def column(name: str) -> pd.Series:
        return cast(pd.Series, df[name]).fillna("").astype(str)

    combined = column(present[0])
    for name in present[1:]:
        combined = combined + " " + column(name)

    # Series.apply is typed as possibly returning a DataFrame, which it only
    # does when the applied function itself returns a Series.
    return cast(pd.Series, combined.apply(clean_text))


# Maps raw category values onto their standard form.
def normalize_categories(values: pd.Series, table: dict, unknown: str = "") -> pd.Series:
    """Map raw category values onto their standard form.

    Used by preprocessing to build labels and by inference to compare
    predictions against ground truth, so both sides read a raw export the same
    way.

    Args:
        values: Raw column values.
        table: Lower-case value -> standard form. Empty leaves values as-is.
        unknown: Standard form for values absent from `table`; when empty they
            become NA so the caller can treat them as unlabelled.
    """
    out = cast(pd.Series, values).astype(str).str.strip()
    if not table:
        return out

    out = out.str.lower().map(table)
    return out.fillna(unknown) if unknown else out
