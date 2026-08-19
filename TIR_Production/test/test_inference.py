#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.test_inference
# Every prediction the system makes goes through `classify`, and its riskiest
# job is routing each row to the classifier belonging to its parent — a row sent
# to the wrong one gets a confidently wrong answer and raises nothing.
#
# These pin the behaviour of all four routing branches, the override path and
# the withholding rules, so a refactor of that function has to preserve them.

import numpy as np
import pytest

from src.inference import classify


HANGER = "cracked weld on pipe hanger"
BOLT = "loose bolt on hanger foundation"
CABLE = "cable damaged inside panel"
LABEL = "missing label on valve body"


# -- the flat level ----------------------------------------------------------


def test_predicts_a_category_for_each_row(bundle):
    out = classify([HANGER, CABLE, LABEL], bundle)
    assert out["pred_process_cat"].tolist() == ["HA Hanger", "EL Electrical", "LP Label"]


def test_confidence_is_a_probability(bundle):
    out = classify([HANGER, CABLE], bundle)
    assert ((out["confidence_process_cat"] > 0) & (out["confidence_process_cat"] <= 1)).all()


# -- routing: the four branches ---------------------------------------------


# A parent with its own model discriminates between its children.
def test_modelled_parent_routes_to_its_own_children(bundle):
    out = classify([HANGER, BOLT], bundle)
    assert out["pred_process_cat"].tolist() == ["HA Hanger", "HA Hanger"]
    assert out["pred_process_sub"].tolist() == ["HAPI Piping", "HABO Bolting"]
    assert out["source_process_sub"].tolist() == ["model", "model"]


# A parent with exactly one possible child needs no model and is certain.
def test_single_child_parent_is_deterministic(bundle):
    out = classify([CABLE], bundle)
    assert out.at[0, "pred_process_cat"] == "EL Electrical"
    assert out.at[0, "pred_process_sub"] == "ELCA Cable"
    assert out.at[0, "confidence_process_sub"] == pytest.approx(1.0)
    assert out.at[0, "source_process_sub"] == "only child of parent"


# A parent with too little data to model falls back, and says so.
def test_unmodelled_parent_falls_back_and_is_flagged(bundle):
    out = classify([LABEL], bundle)
    assert out.at[0, "pred_process_cat"] == "LP Label"
    assert out.at[0, "pred_process_sub"] == "LPAS Assembly"
    assert out.at[0, "source_process_sub"] == "no model for parent"
    assert bool(out.at[0, "review_process_sub"]) is True


# Whatever the child answers, it belongs under the parent that was predicted.
def test_child_always_belongs_to_its_parent(bundle):
    out = classify([HANGER, BOLT, CABLE, LABEL], bundle)
    under = {
        "HA Hanger": {"HAPI Piping", "HABO Bolting"},
        "EL Electrical": {"ELCA Cable"},
        "LP Label": {"LPAS Assembly"},
    }
    for _, row in out.iterrows():
        assert row["pred_process_sub"] in under[row["pred_process_cat"]]


# -- withholding -------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "nan"])
def test_blank_text_is_withheld_at_every_level(bundle, text):
    out = classify([text], bundle)
    for target in ("process_cat", "process_sub"):
        assert out.at[0, f"pred_{target}"] == ""
        assert out.at[0, f"confidence_{target}"] == 0.0
        assert out.at[0, f"source_{target}"] == "no description"
        assert bool(out.at[0, f"review_{target}"]) is True


# A blank row must not disturb the rows either side of it.
def test_a_blank_row_does_not_affect_its_neighbours(bundle):
    out = classify([HANGER, "", CABLE], bundle)
    assert out["pred_process_cat"].tolist() == ["HA Hanger", "", "EL Electrical"]
    assert out["pred_process_sub"].tolist() == ["HAPI Piping", "", "ELCA Cable"]


# -- reviewer overrides ------------------------------------------------------


def test_override_replaces_the_prediction_and_clears_the_flag(bundle):
    out = classify([CABLE], bundle, {"process_cat": ["LP Label"]})
    assert out.at[0, "pred_process_cat"] == "LP Label"
    assert out.at[0, "confidence_process_cat"] == pytest.approx(1.0)
    assert out.at[0, "source_process_cat"] == "confirmed by reviewer"
    assert bool(out.at[0, "review_process_cat"]) is False


# The whole point of confirming a category: the level below is re-chosen inside
# the confirmed one, not the predicted one.
def test_override_reroutes_the_child(bundle):
    without = classify([CABLE], bundle)
    with_override = classify([CABLE], bundle, {"process_cat": ["HA Hanger"]})
    assert without.at[0, "pred_process_sub"] == "ELCA Cable"
    assert with_override.at[0, "pred_process_sub"] in {"HAPI Piping", "HABO Bolting"}


# An empty override leaves that row to the model.
def test_empty_override_is_ignored(bundle):
    out = classify([HANGER, CABLE], bundle, {"process_cat": ["", "LP Label"]})
    assert out.at[0, "pred_process_cat"] == "HA Hanger"
    assert out.at[1, "pred_process_cat"] == "LP Label"


# -- review flags ------------------------------------------------------------


# A parent the model was unsure of makes its child unsure too, because a wrong
# parent guarantees a wrong child.
def test_review_flag_is_inherited_from_the_parent(bundle):
    entry = bundle["targets"]["process_cat"]
    original = entry["threshold"]
    entry["threshold"] = 1.1          # force every parent below the bar
    try:
        out = classify([HANGER], bundle)
        assert bool(out.at[0, "review_process_cat"]) is True
        assert bool(out.at[0, "review_process_sub"]) is True
    finally:
        entry["threshold"] = original


# ...unless a person confirmed the child themselves.
def test_a_confirmed_child_does_not_inherit_the_flag(bundle):
    entry = bundle["targets"]["process_cat"]
    original = entry["threshold"]
    entry["threshold"] = 1.1
    try:
        out = classify([HANGER], bundle, {"process_sub": ["HABO Bolting"]})
        assert bool(out.at[0, "review_process_cat"]) is True
        assert bool(out.at[0, "review_process_sub"]) is False
    finally:
        entry["threshold"] = original


# -- alternates --------------------------------------------------------------


# Where a level reports a ranked list, the alternates exclude the answer itself
# and stay inside the same parent.
def test_alternates_are_returned_and_stay_within_the_parent(bundle):
    out = classify([HANGER], bundle)
    alternates = [a for a in out.at[0, "alt_process_sub"].split(", ") if a]
    assert alternates == ["HABO Bolting"]
    assert out.at[0, "pred_process_sub"] not in alternates


def test_no_alternates_column_when_top_k_is_one(bundle):
    assert "alt_process_cat" not in classify([HANGER], bundle).columns
