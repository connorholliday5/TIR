#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.conftest
# A small but genuine trained bundle, so the routing tests exercise the real
# loading and scoring path rather than a stand-in for it.
#
# The taxonomy is shaped to reach every branch in the child router: one parent
# with several children and a model of its own, one parent with exactly one
# possible child, and one parent with too little data to have been modelled at
# all.

import json

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


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """A trained two-level bundle, loaded through the real `load_bundle`."""
    root = tmp_path_factory.mktemp("bundle")
    models, data = root / "models", root / "data"
    models.mkdir()
    data.mkdir()

    word = TfidfVectorizer().fit(CORPUS)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 4)).fit(CORPUS)
    joblib.dump(word, models / "tfidf_word.pkl")
    joblib.dump(char, models / "tfidf_char.pkl")
    X = build_features(CORPUS, word, char)

    (data / "label_map_process_cat.json").write_text(
        json.dumps({str(k): {"name": v} for k, v in CATS.items()})
    )
    (data / "label_map_process_sub.json").write_text(
        json.dumps({str(k): {"name": v} for k, v in SUBS.items()})
    )

    cat_dir = models / "process_cat"
    cat_dir.mkdir()
    joblib.dump(
        CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3).fit(X, CAT_LABELS),
        cat_dir / "svm.pkl",
    )
    (cat_dir / "ensemble.json").write_text(json.dumps({"weights": {"svm": 1.0}}))

    # Only HA Hanger (id 0) gets a model of its own.
    sub_dir = models / "process_sub"
    sub_dir.mkdir()
    ha = sub_dir / "0"
    ha.mkdir()
    rows = np.array(CAT_LABELS) == 0
    joblib.dump(
        CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3).fit(
            X[rows], np.array(SUB_LABELS)[rows]
        ),
        ha / "svm.pkl",
    )
    (ha / "ensemble.json").write_text(json.dumps({"weights": {"svm": 1.0}}))

    (sub_dir / "routing.json").write_text(json.dumps({
        "parent": "process_cat",
        "deterministic": {"EL Electrical": "ELCA Cable"},
        "fallback": {"HA Hanger": "HAPI Piping", "LP Label": "LPAS Assembly"},
        "num_labels": len(SUBS),
    }))

    import src.inference as inference
    targets = {
        "process_cat": {"column": "process_cat", "review_threshold": 0.5},
        "process_sub": {
            "column": "process_sub", "parent": "process_cat",
            "review_threshold": 0.4, "top_k": 2,
        },
    }
    original = inference.TARGETS
    inference.TARGETS = targets
    inference.CONFIG["targets"] = targets
    try:
        yield inference.load_bundle(models, data)
    finally:
        inference.TARGETS = original
        inference.CONFIG["targets"] = original
