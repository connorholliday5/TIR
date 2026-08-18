#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.columns
# Maps the column names of a raw QPS export onto the canonical names the
# pipeline works in.
#
# The TIR team exports the same fields under different headings depending on
# which report produced the file: "Description 1", "DESCRIPTION_ONE" and "Item
# Description 1" are all the same field.  Every module reads canonical names,
# and this is the only place that knows what a given workbook calls them.
#
# The alias table lives in config.json, so a fourth export layout is a
# configuration change rather than a code change.

import re
from typing import Dict, List, Sequence

import pandas as pd


# Reduces a heading to the form used for matching.
def _key(name: str) -> str:
    """Fold a column heading to its comparison key.

    Case, surrounding whitespace, underscores and repeated spaces all vary
    between exports of the same field ("DESCRIPTION_ONE" against "Description
    One"), and none of them carry meaning, so they are normalised away rather
    than enumerated as separate aliases.
    """
    folded = re.sub(r"[\s_]+", " ", str(name).strip().lower())
    return folded.strip()


# Finds the actual column backing each canonical name.
def resolve_columns(
    df: pd.DataFrame, aliases: Dict[str, Sequence[str]]
) -> Dict[str, str]:
    """Return {canonical name: the column of `df` that holds it}.

    Canonical names the frame does not carry are simply absent from the
    result, so callers can distinguish "this export has no Description 2" from
    "Description 2 is empty".

    Args:
        df: A raw export.
        aliases: Canonical name -> the headings that stand for it, best first.

    Returns:
        The mapping, containing only the canonical names actually present.
    """
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
    df: pd.DataFrame, aliases: Dict[str, Sequence[str]]
) -> pd.DataFrame:
    """Return a copy of `df` with recognised columns renamed to canonical form.

    Columns no alias claims are left exactly as they were, so an uploaded file
    keeps every field it arrived with and the batch tab can hand them back.

    A canonical name that already exists as a literal column is not renamed
    over: that would silently drop one of the two.
    """
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
    df: pd.DataFrame, required: Sequence[str], aliases: Dict[str, Sequence[str]]
) -> List[str]:
    """Return the required canonical names that `df` cannot supply."""
    resolved = resolve_columns(df, aliases)
    return [name for name in required if name not in resolved and name not in df.columns]
