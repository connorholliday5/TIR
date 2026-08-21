#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.conftest
# Small but genuine trained bundles, so the routing tests exercise the real
# loading and scoring path rather than a stand-in for it.
#
# `bundle` is shaped to reach every branch in the child router: one parent with
# several children and a model of its own, one parent with exactly one possible
# child, and one parent with too little data to have been modelled at all.
#
# `branched_bundle` covers the other routing decision — whether a family of
# codes applies to the record at all.

import json
from contextlib import contextmanager

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.models import build_features


CATS = {0: "HA Hanger", 1: "EL Electrical", 2: "LP Label"}
SUBS = {
    0: "HAPI Piping",     # under HA, modelled
    1: "HABO Bolting",    # under HA, modelled
    2: "ELCA Cable",      # under EL, the only child
    3: "LPAS Assembly",   # under LP, no model — fallback only
}

CORPUS = (
    ["cracked weld on pipe hanger"] * 6      # HA / HAPI
    + ["loose bolt on hanger foundation"] * 6  # HA / HABO
    + ["cable damaged inside panel"] * 6       # EL / ELCA
    + ["missing label on valve body"] * 3      # LP / LPAS
)
CAT_LABELS = [0] * 12 + [1] * 6 + [2] * 3
SUB_LABELS = [0] * 6 + [1] * 6 + [2] * 6 + [3] * 3

PROGRAM_CATS = {0: "OHS Health & Safety", 1: "EP Equipment Protection"}

# Records carrying Program codes, and records that carry none.  Program codes
# appear on 72.9% of SURV records and 6.1% of TIRs, so a model trained on the
# coded rows alone has never seen the majority of what it will be asked about.
IN_BRANCH = (
    ["exposed plug left uncapped near walkway"] * 5   # OHS
    + ["guard rail missing from equipment platform"] * 5  # EP
)
OUT_OF_BRANCH = ["cracked weld on pipe hanger"] * 10
PROGRAM_LABELS = [0] * 5 + [1] * 5


# Points the inference module at a test taxonomy for the duration of a block.
@contextmanager
def configured(targets: dict, branches: dict):
    """Swap in `targets` and `branches`, restoring both afterwards.

    `load_bundle` reads the configuration through module-level names rather
    than arguments, so a fixture has to substitute them.  Restoring in a
    `finally` matters because the fixtures are session-scoped: a leaked
    taxonomy would be inherited by whichever test ran next.
    """
    import src.inference as inference

    was_targets, was_branches = inference.TARGETS, inference.BRANCHES
    inference.TARGETS = targets
    inference.BRANCHES = branches
    inference.CONFIG["targets"] = targets
    inference.CONFIG["branches"] = branches
    try:
        yield
    finally:
        inference.TARGETS = was_targets
        inference.BRANCHES = was_branches
        inference.CONFIG["targets"] = was_targets
        inference.CONFIG["branches"] = was_branches


# Fits a calibrated model and writes it where the loader expects to find it.
def _save_model(directory, X, y):
    """Save a fitted classifier and its single-member ensemble spec."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3).fit(X, y),
        directory / "svm.pkl",
    )
    (directory / "ensemble.json").write_text(json.dumps({"weights": {"svm": 1.0}}))


# Fits the shared vectorizers over a corpus and returns them with the matrix.
def _vectorize(models, corpus):
    word = TfidfVectorizer().fit(corpus)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 4)).fit(corpus)
    joblib.dump(word, models / "tfidf_word.pkl")
    joblib.dump(char, models / "tfidf_char.pkl")
    return build_features(corpus, word, char)


# Creates the models/ and data/ pair a bundle is loaded from.
def _workspace(tmp_path_factory, name):
    root = tmp_path_factory.mktemp(name)
    models, data = root / "models", root / "data"
    models.mkdir()
    data.mkdir()
    return models, data


# Writes the id -> name map for one target.
def _write_label_map(data, target, names):
    (data / f"label_map_{target}.json").write_text(
        json.dumps({str(k): {"name": v} for k, v in names.items()})
    )


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """A trained two-level bundle, loaded through the real `load_bundle`."""
    import src.inference as inference

    models, data = _workspace(tmp_path_factory, "bundle")
    X = _vectorize(models, CORPUS)

    _write_label_map(data, "process_cat", CATS)
    _write_label_map(data, "process_sub", SUBS)

    _save_model(models / "process_cat", X, CAT_LABELS)

    # Only HA Hanger (id 0) gets a model of its own.
    rows = np.array(CAT_LABELS) == 0
    _save_model(models / "process_sub" / "0", X[rows], np.array(SUB_LABELS)[rows])

    (models / "process_sub" / "routing.json").write_text(json.dumps({
        "parent": "process_cat",
        "deterministic": {"EL Electrical": "ELCA Cable"},
        "fallback": {"HA Hanger": "HAPI Piping", "LP Label": "LPAS Assembly"},
        "num_labels": len(SUBS),
    }))

    # The observed parent/child pairs, as preprocessing records them.  What a
    # child may be given is read from here, so a fixture without it would leave
    # the candidate list silently empty.
    (data / "hierarchy.json").write_text(json.dumps({
        "process_sub": {
            "HA Hanger": ["HABO Bolting", "HAPI Piping"],
            "EL Electrical": ["ELCA Cable"],
            "LP Label": ["LPAS Assembly"],
        },
    }))

    targets = {
        "process_cat": {"column": "process_cat", "review_threshold": 0.5},
        "process_sub": {
            "column": "process_sub", "parent": "process_cat",
            "review_threshold": 0.4, "top_k": 2,
        },
    }
    # No branch here: these targets are always answered, so the fixture keeps
    # the child-routing tests to the one thing they are about.
    with configured(targets, branches={}):
        yield inference.load_bundle(models, data)


@pytest.fixture(scope="session")
def branched_bundle(tmp_path_factory):
    """A bundle whose target sits behind a branch that often does not apply.

    The Program model is fitted on the coded rows alone, exactly as training
    does it, so it has never seen a hanger TIR and will answer one as
    confidently as anything else.  Whether that answer reaches the output is
    the branch's decision, which is what these tests are about.
    """
    import src.inference as inference

    models, data = _workspace(tmp_path_factory, "branched")
    corpus = IN_BRANCH + OUT_OF_BRANCH
    applies = [1] * len(IN_BRANCH) + [0] * len(OUT_OF_BRANCH)
    X = _vectorize(models, corpus)

    _write_label_map(data, "program_cat", PROGRAM_CATS)

    in_branch = np.array(applies) == 1
    _save_model(models / "program_cat", X[in_branch], PROGRAM_LABELS)
    _save_model(models / "branch_program", X, applies)

    targets = {
        "program_cat": {
            "column": "program_cat", "branch": "program", "review_threshold": 0.5,
        },
    }
    branches = {
        "program": {
            "anchor": "program_cat", "title": "Program codes",
            "review_threshold": 0.5,
        },
    }
    with configured(targets, branches):
        yield inference.load_bundle(models, data)
