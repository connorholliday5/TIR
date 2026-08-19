#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.test_reports
# Covers the coder-consistency measure, which is the figure every accuracy
# number in the study is reported against — so it is worth pinning to
# hand-checked cases rather than trusting it to stay right through a refactor.

import pandas as pd
import pytest

from src.reports import consistency, threshold_for, verdict
import numpy as np


# Two repeated descriptions, one coded consistently and one not, plus a
# description that appears only once and must be ignored entirely.
def test_agreement_on_a_hand_checked_case():
    df = pd.DataFrame({
        "repeat_key": ["a", "a", "a", "b", "b", "c"],
        "cat":        ["X", "X", "Y", "Z", "Z", "W"],
    })
    result = consistency(df, "cat")
    assert result["groups"] == 2          # a and b repeat; c does not
    assert result["rows"] == 5            # the single c row is not counted
    assert result["conflicting"] == 1     # only a was given two different codes
    assert result["agreement"] == pytest.approx(4 / 5)


# Perfect agreement is the ceiling, and must not be reported as anything less.
def test_total_agreement_scores_one():
    df = pd.DataFrame({"repeat_key": ["a", "a"], "cat": ["X", "X"]})
    result = consistency(df, "cat")
    assert result["agreement"] == pytest.approx(1.0)
    assert result["conflicting"] == 0


# With nothing repeated there is no ceiling to report, and NaN says so rather
# than a fabricated 100%.
def test_no_repeats_reports_unknown():
    df = pd.DataFrame({"repeat_key": ["a", "b"], "cat": ["X", "Y"]})
    result = consistency(df, "cat")
    assert result["groups"] == 0
    assert result["agreement"] != result["agreement"]  # NaN


# Blank codes are absent, not a category of their own.
def test_uncoded_rows_are_excluded():
    df = pd.DataFrame({
        "repeat_key": ["a", "a", "a"],
        "cat":        ["X", "X", None],
    })
    assert consistency(df, "cat")["rows"] == 2


# An unmeasurable field is described as unknown, not judged.
def test_verdict_handles_missing_agreement():
    assert "unknown" in verdict(float("nan"), 10, 0)


@pytest.mark.parametrize("agreement,expected", [
    (0.95, "sufficient"), (0.80, "marginal"), (0.50, "insufficient"),
])
def test_verdict_bands(agreement, expected):
    assert expected in verdict(agreement, 10, 0)


# The coverage search returns the *lowest* qualifying threshold, since that
# leaves the most TIRs coded automatically.
def test_threshold_search_prefers_wider_coverage():
    confidence = np.linspace(0.0, 1.0, 500)
    correct = confidence > 0.4          # everything above 0.4 is right
    found = threshold_for(confidence, correct, 0.95)
    assert found is not None
    threshold, coverage, precision = found
    assert precision >= 0.95
    assert threshold < 0.7              # not driven needlessly high


# A target no threshold can reach is reported as unreachable, not approximated.
def test_unreachable_precision_returns_none():
    confidence = np.linspace(0.5, 1.0, 500)
    correct = np.zeros(500, dtype=bool)
    assert threshold_for(confidence, correct, 0.95) is None
