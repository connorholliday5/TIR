#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.test_columns
# Covers the alias layer: the same field arrives under a different heading in
# every QPS export, and everything downstream assumes canonical names.

import pandas as pd
import pytest

from src.columns import canonicalize, missing_columns, resolve_columns


ALIASES = {
    "description_1": ["Description 1", "DESCRIPTION_ONE", "Item Description 1"],
    "process_cat": ["Process Cat", "PROCESS_CATEGORY", "Item Process Category"],
}


# Each of the three export layouts resolves to the same canonical field.
@pytest.mark.parametrize("heading", ["Description 1", "DESCRIPTION_ONE", "Item Description 1"])
def test_every_layout_resolves(heading):
    df = pd.DataFrame({heading: ["x"]})
    assert resolve_columns(df, ALIASES)["description_1"] == heading


# Case, underscores and repeated spaces vary between exports and carry no
# meaning, so they must not have to be enumerated as separate aliases.
@pytest.mark.parametrize("heading", ["description_one", "Description One", "DESCRIPTION  ONE"])
def test_matching_ignores_case_and_separators(heading):
    df = pd.DataFrame({heading: ["x"]})
    assert resolve_columns(df, ALIASES)["description_1"] == heading


# A field the file does not carry is absent rather than blank, so callers can
# tell "no such column" from "column is empty".
def test_absent_field_is_not_reported():
    df = pd.DataFrame({"Description 1": ["x"]})
    assert "process_cat" not in resolve_columns(df, ALIASES)


# Columns nothing claims must survive: the batch tab hands the uploaded file
# back with every field it arrived with.
def test_unclaimed_columns_are_untouched():
    df = pd.DataFrame({"DESCRIPTION_ONE": ["x"], "Job Order": ["7"]})
    out = canonicalize(df, ALIASES)
    assert list(out.columns) == ["description_1", "Job Order"]


# Renaming onto a name the frame already uses would drop one of the two.
def test_existing_canonical_column_is_not_overwritten():
    df = pd.DataFrame({"description_1": ["kept"], "DESCRIPTION_ONE": ["other"]})
    out = canonicalize(df, ALIASES)
    assert out["description_1"].tolist() == ["kept"]
    assert "DESCRIPTION_ONE" in out.columns


# Validation reports the canonical name, which is what the user configured.
def test_missing_columns_names_the_canonical_field():
    df = pd.DataFrame({"Unrelated": ["x"]})
    assert missing_columns(df, ["description_1"], ALIASES) == ["description_1"]
