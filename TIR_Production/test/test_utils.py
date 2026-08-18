#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.test_utils
# Covers text assembly and the ensemble arithmetic behind the number shown to
# a coder as "confidence".

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    build_text_series, clean_text, combine_ensemble_scores,
    ensemble_score_matrix, is_blank_text, top_k_from_scores,
)


# Not every export carries a Description 2; a file lacking one should still
# classify on what it does have rather than be rejected.
def test_absent_text_columns_are_skipped():
    df = pd.DataFrame({"description_1": ["Bolt  loose"], "doc_title": ["T"]})
    out = build_text_series(df, ["description_1", "description_2", "doc_title"])
    assert out.tolist() == ["bolt loose t"]


# Losing every text column is a different situation and must be an error.
def test_no_text_column_at_all_raises():
    with pytest.raises(ValueError):
        build_text_series(pd.DataFrame({"other": ["x"]}), ["description_1"])


# A missing value must not become the token "nan" appended to every row.
def test_missing_values_do_not_leak_a_token():
    df = pd.DataFrame({"description_1": ["weld"], "description_2": [None]})
    assert build_text_series(df, ["description_1", "description_2"]).iloc[0] == "weld"


# Runs of whitespace carry no meaning but character n-grams can see them.
def test_clean_text_collapses_whitespace():
    assert clean_text("  A\t\tB\n C ") == "a b c"


@pytest.mark.parametrize("value", ["", "   ", "nan", "None"])
def test_blank_text_is_recognised(value):
    assert is_blank_text(value)


# The blended score must be a probability, so it can be compared against a
# review threshold and read as "at least this likely".  The previous form
# one-hot encoded the SVM's pick at a flat 0.40, which put a floor under the
# winner and a ceiling over every rival.
def test_blended_confidence_is_a_probability():
    probas = {"svm": np.array([[0.1, 0.7, 0.2]]), "xgb": np.array([[0.6, 0.3, 0.1]])}
    _, confidence = combine_ensemble_scores(probas, {"svm": 0.5, "xgb": 0.5}, 3)
    assert confidence[0] == pytest.approx(0.5)
    assert 0.0 <= confidence[0] <= 1.0


# A single member is used as-is rather than diluted.
def test_single_member_passes_through():
    probas = {"svm": np.array([[0.1, 0.7, 0.2]])}
    ids, confidence = combine_ensemble_scores(probas, {"svm": 1.0, "xgb": 0.0}, 3)
    assert ids[0] == 1 and confidence[0] == pytest.approx(0.7)


# Weights are renormalised, so they need not be supplied summing to one.
def test_weights_are_renormalised():
    probas = {"svm": np.array([[0.2, 0.8]]), "lr": np.array([[0.2, 0.8]])}
    matrix = ensemble_score_matrix(probas, {"svm": 2.0, "lr": 2.0}, 2)
    assert matrix[0].sum() == pytest.approx(1.0)


# Weighting every member to zero cannot silently predict class 0.
def test_no_active_member_is_an_error():
    with pytest.raises(ValueError):
        ensemble_score_matrix({"svm": np.array([[0.5, 0.5]])}, {"svm": 0.0}, 2)


# The deepest category level is offered as a short ranked list, best first.
def test_top_k_is_ordered_best_first():
    ids, scores = top_k_from_scores(np.array([[0.1, 0.7, 0.2]]), 2)
    assert ids[0].tolist() == [1, 2]
    assert scores[0].tolist() == pytest.approx([0.7, 0.2])


# Asking for more alternatives than there are categories must not fail.
def test_top_k_larger_than_label_space():
    ids, _ = top_k_from_scores(np.array([[0.4, 0.6]]), 5)
    assert ids.shape == (1, 2)
