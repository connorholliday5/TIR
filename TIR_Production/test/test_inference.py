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

from src.inference import _consult_llm, _score, classify
from src.llm import FakeBackend
from src.models import build_features

from conftest import IN_BRANCH, OUT_OF_BRANCH, PROGRAM_CATS


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


# -- branch routing ----------------------------------------------------------
#
# A branch decides whether a family of codes applies to the record at all.
# Process codes are a TIR field and Program codes a SURV field, so most records
# carry one branch and not the other; the in-branch models never saw a record
# outside their branch and cannot be trusted to recognise one.


# The case the branch exists for: a record that carries no Program codes.
def test_branch_withholds_a_record_outside_it(branched_bundle):
    out = classify(OUT_OF_BRANCH[:1], branched_bundle)

    assert out.loc[0, "pred_program_cat"] == ""
    assert out.loc[0, "confidence_program_cat"] == 0.0
    assert out.loc[0, "source_program_cat"] == "no program codes on this record"


# A withheld branch is not a doubtful answer, so it is not queued for review.
def test_withheld_branch_is_not_flagged_for_review(branched_bundle):
    out = classify(OUT_OF_BRANCH[:1], branched_bundle)

    assert not out.loc[0, "review_program_cat"]


# The branch must not swallow the records it does apply to.
def test_branch_answers_a_record_inside_it(branched_bundle):
    out = classify(IN_BRANCH[:1], branched_bundle)

    assert out.loc[0, "pred_program_cat"] in set(PROGRAM_CATS.values())
    assert out.loc[0, "confidence_program_cat"] > 0.0


# Both sides in one call, since the branch is evaluated for the batch at once.
def test_branch_separates_a_mixed_batch(branched_bundle):
    out = classify([IN_BRANCH[0], OUT_OF_BRANCH[0], IN_BRANCH[-1]], branched_bundle)

    assert out.loc[0, "pred_program_cat"] != ""
    assert out.loc[1, "pred_program_cat"] == ""
    assert out.loc[2, "pred_program_cat"] != ""


# Without the branch the same model answers the same record confidently, which
# is what makes the branch worth having rather than a redundant safety net.
def test_the_model_alone_would_have_answered_the_outside_record(branched_bundle):
    entry = branched_bundle["targets"]["program_cat"]
    X = build_features(OUT_OF_BRANCH[:1], branched_bundle["tfidf_word"],
                       branched_bundle["tfidf_char"])

    scores = _score(entry, X, len(entry["id2name"]))

    assert scores.max() > 0.5


# A blank description has nothing to read, and that rule outranks the branch.
def test_blank_text_is_withheld_whatever_the_branch_says(branched_bundle):
    out = classify(["   "], branched_bundle)

    assert out.loc[0, "pred_program_cat"] == ""
    assert out.loc[0, "source_program_cat"] == "no description"


# A coder's explicit answer outranks the branch: they can see the record.
def test_an_override_survives_a_branch_that_does_not_apply(branched_bundle):
    out = classify(
        OUT_OF_BRANCH[:1], branched_bundle,
        overrides={"program_cat": ["OHS Health & Safety"]},
    )

    assert out.loc[0, "pred_program_cat"] == "OHS Health & Safety"
    assert out.loc[0, "source_program_cat"] == "confirmed by reviewer"


# -- language-model fallback -------------------------------------------------
#
# The model is consulted only on rows the classifiers flagged for review. The
# classifiers already match how consistently people code these fields, so a
# confident row has nothing to gain and 40,000 records a year makes asking
# about every one of them expensive.


# The whole point of the fallback: a flagged row gets a second opinion.
def test_a_flagged_row_is_offered_to_the_model(bundle):
    backend = FakeBackend({"missing label": "HAPI Piping — it names a hanger"})

    out = classify(["missing label on valve body"], bundle, backend=backend)

    assert backend.prompts, "a flagged row should have reached the model"


# A confident row must not: that is what keeps the batch path affordable.
def test_a_confident_row_never_reaches_the_model(bundle):
    backend = FakeBackend()

    out = classify(["cracked weld on pipe hanger"], bundle, backend=backend)

    confident = not out.loc[0, "review_process_cat"]
    assert confident
    assert all("Process Cat" not in p for p in backend.prompts)


# An answer outside the candidate list leaves the classifier's answer standing.
def test_an_invalid_suggestion_changes_nothing(bundle):
    backend = FakeBackend({"missing label": "ZZTOP Invented"})

    baseline = classify(["missing label on valve body"], bundle)
    out = classify(["missing label on valve body"], bundle, backend=backend)

    assert out.loc[0, "pred_process_cat"] == baseline.loc[0, "pred_process_cat"]
    assert out.loc[0, "source_process_cat"] == baseline.loc[0, "source_process_cat"]


# Without a backend the frame is exactly what it always was — no stray column.
def test_no_backend_leaves_the_frame_unchanged(bundle):
    out = classify(["missing label on valve body"], bundle)

    assert not [c for c in out.columns if c.startswith("rationale_")]


# A suggestion the model supplies replaces the answer and says where it came
# from, so a coder is never shown a machine's guess as if a classifier made it.
# Driven directly rather than through a threshold, so the row under test is
# flagged by construction and the assertion cannot quietly stop running.
def test_an_accepted_suggestion_replaces_the_answer_and_is_attributed(bundle):
    entry = bundle["targets"]["process_cat"]
    chosen = list(entry["id2name"].values())[0]
    backend = FakeBackend({"valve body": f"{chosen} — best fit"})

    out = classify(["missing label on valve body"], bundle)
    out.loc[0, "review_process_cat"] = True

    _consult_llm(out, "process_cat", entry, bundle,
                 ["missing label on valve body"], backend, {})

    assert out.loc[0, "pred_process_cat"] == chosen
    assert out.loc[0, "source_process_cat"] == "language model"
    assert "best fit" in out.loc[0, "rationale_process_cat"]


# A child's suggestions are drawn from its parent's own children, so the model
# cannot produce a pairing the taxonomy does not contain.
def test_suggestions_for_a_child_are_limited_to_its_parents_children(bundle):
    entry = bundle["targets"]["process_sub"]
    backend = FakeBackend()

    out = classify(["cracked weld on pipe hanger"], bundle)
    out.loc[0, "review_process_sub"] = True

    _consult_llm(out, "process_sub", entry, bundle,
                 ["cracked weld on pipe hanger"], backend, {})

    offered = backend.prompts[0]
    assert "HAPI Piping" in offered
    assert "ELCA Cable" not in offered
