#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.test_models
# Covers text assembly and the ensemble arithmetic behind the number shown to
# a coder as "confidence".

import numpy as np
import pandas as pd
import pytest

from src.data import build_text_series, clean_text, is_blank_text
from src.models import (
    combine_ensemble_scores, ensemble_score_matrix, top_k_from_scores,
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


# -- correction reuse --------------------------------------------------------
#
# Corrections are reapplied to near-duplicate wordings using the classifiers'
# own TF-IDF features.  The threshold has to be loose enough that an edit which
# leaves a TIR the same TIR still matches, and tight enough that a genuine
# rewording goes back to the model.

import pytest as _pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feedback import CorrectionIndex


@_pytest.fixture(scope="module")
def index():
    corpus = [
        "cracked weld on pipe hanger", "cable damaged inside panel",
        "missing label on valve body", "loose bolt on foundation",
        "paint missing from bracket", "hanger fit not like drawing",
    ]
    word = TfidfVectorizer(ngram_range=(1, 2)).fit(corpus)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit(corpus)
    return CorrectionIndex(
        ["cracked weld on pipe hanger"], ["HA HANGER"], word, char,
    )


# An exact repeat is answered with full confidence.
def test_exact_match_is_certain(index):
    assert index.lookup("cracked weld on pipe hanger") == ("HA HANGER", 1.0)


# Whitespace and case are not meaningful differences.
def test_match_survives_reformatting(index):
    label, _ = index.lookup("  Cracked   WELD on pipe hanger ")
    assert label == "HA HANGER"


# A wording nothing was corrected for is left to the model.
def test_unrelated_text_does_not_match(index):
    assert index.lookup("missing label on valve body") == (None, 0.0)


# A blank lookup must not be anyone's nearest neighbour.
def test_blank_lookup_does_not_match(index):
    assert index.lookup("") == (None, 0.0)


# An empty index answers nothing rather than failing.
def test_empty_index_is_safe():
    assert CorrectionIndex([], []).lookup("anything") == (None, 0.0)
    assert len(CorrectionIndex([], [])) == 0


# A threshold above 1.0 is the documented way to accept exact matches only.
def test_threshold_above_one_allows_exact_only():
    idx = CorrectionIndex(["bolt loose"], ["FA FASTENER"], threshold=1.5)
    assert idx.lookup("bolt loose")[0] == "FA FASTENER"
    assert idx.lookup("bolt is loose") == (None, 0.0)


# -- blend weighting ---------------------------------------------------------
#
# Two models that both clear the gate are not therefore equally good. Splitting
# evenly is the obvious choice and the wrong one — on Process Category the
# second model scored 2.4 points above the first and was trusted the same
# amount.

from src.models import search_blend_weights, weight_grid


# Every candidate weighting is a distribution.
def test_weight_grid_sums_to_one():
    for candidate in weight_grid(["a", "b"], step=0.25):
        assert sum(candidate.values()) == pytest.approx(1.0)


def test_weight_grid_covers_the_endpoints():
    grid = list(weight_grid(["a", "b"], step=0.5))
    assert {"a": 1.0, "b": 0.0} in grid
    assert {"a": 0.0, "b": 1.0} in grid


# A single member is trusted entirely; there is nothing to weigh it against.
def test_single_member_takes_all_the_weight():
    probas = {"svm": np.array([[0.9, 0.1], [0.2, 0.8]])}
    weights, _, _ = search_blend_weights(probas, np.array([0, 1]), 2)
    assert weights == {"svm": 1.0}


# Where one member is right and the other wrong, the search should stop
# trusting the wrong one rather than average them.
def test_search_moves_weight_towards_the_better_member():
    truth = np.array([0, 1, 0, 1])
    good = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    bad = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    weights, best, even = search_blend_weights({"good": good, "bad": bad}, truth, 2)
    assert weights["good"] > weights["bad"]
    assert best >= even


# The returned weighting is the one whose score is reported.
def test_reported_score_matches_the_chosen_weighting():
    truth = np.array([0, 1, 0, 1])
    a = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]])
    b = np.array([[0.6, 0.4], [0.4, 0.6], [0.9, 0.1], [0.1, 0.9]])
    weights, best, _ = search_blend_weights({"a": a, "b": b}, truth, 2)
    from src.models import ensemble_score_matrix
    from sklearn.metrics import f1_score
    predicted = ensemble_score_matrix({"a": a, "b": b}, weights, 2).argmax(axis=1)
    assert f1_score(truth, predicted, average="macro", zero_division=0) == pytest.approx(best)


# Two members of equal quality should not be disturbed off an even split.
def test_equal_members_stay_even():
    truth = np.array([0, 1, 0, 1])
    same = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    weights, best, even = search_blend_weights({"a": same, "b": same.copy()}, truth, 2)
    assert best == pytest.approx(even)
    assert weights == {"a": 0.5, "b": 0.5}
