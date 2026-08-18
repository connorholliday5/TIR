#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.inference
# Loads the trained artifacts and turns cleaned text into predictions.
#
# The batch pipeline, the web UI and the HTTP API all classify through here so
# they cannot drift apart on feature construction, ensemble weighting or the
# rules that decide when an answer is withheld.

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

try:  # modules under src/
    from src.config import (
        CONFIG, DATA_DIR, MODEL_DIR, PRIMARY_TARGET, TARGETS,
        review_threshold, target_title,
    )
    from src.data import is_blank_text
    from src.models import (
        build_features, ensemble_score_matrix, expand_proba,
        top_k_from_scores, verify_model_hash,
    )
except ImportError:  # flat layout, modules at the repository root
    from config import (
        CONFIG, DATA_DIR, MODEL_DIR, PRIMARY_TARGET, TARGETS,
        review_threshold, target_title,
    )
    from data import is_blank_text
    from models import (
        build_features, ensemble_score_matrix, expand_proba,
        top_k_from_scores, verify_model_hash,
    )


# Loads the models saved for one directory, honouring its ensemble.json.
def _load_member_models(directory: Path) -> dict:
    """Return {"models": …, "weights": …} for a target or per-parent folder."""
    spec_path = directory / "ensemble.json"
    weights = {"svm": 1.0}
    if spec_path.is_file():
        weights = json.loads(spec_path.read_text()).get("weights", weights)

    models = {}
    for name in weights:
        if name == "xgb":
            path = directory / "xgb.json"
            if path.is_file():
                import xgboost as xgb_lib
                booster = xgb_lib.Booster()
                booster.load_model(str(path))
                models["xgb"] = booster
        else:
            path = directory / f"{name}.pkl"
            if path.is_file():
                verify_model_hash(path)
                models[name] = joblib.load(path)

    return {"models": models, "weights": {k: v for k, v in weights.items() if k in models}}


# Scores one bundle of models over a feature matrix.
def _score(member: dict, X, num_labels: int) -> np.ndarray:
    """Return the blended (n_rows, num_labels) probability matrix."""
    probas = {}
    for name, model in member["models"].items():
        if name == "xgb":
            import xgboost as xgb_lib
            probas[name] = expand_proba(
                model.predict(xgb_lib.DMatrix(X)), np.arange(num_labels), num_labels
            )
        else:
            probas[name] = expand_proba(model.predict_proba(X), model.classes_, num_labels)

    return ensemble_score_matrix(probas, member["weights"], num_labels)


# Loads every artifact needed to classify.
def load_bundle(model_dir: Path = MODEL_DIR, data_dir: Path = DATA_DIR) -> dict:
    """Verify and load the shared vectorizers and each target's classifiers.

    Raises:
        FileNotFoundError: If an artifact is missing or fails its SHA-256 check.
    """
    missing = [
        n for n in ("tfidf_word.pkl", "tfidf_char.pkl") if not (model_dir / n).is_file()
    ]
    for target in TARGETS:
        if not (data_dir / f"label_map_{target}.json").is_file():
            missing.append(f"data/label_map_{target}.json")
    if missing:
        raise FileNotFoundError(", ".join(missing))

    # Integrity first: joblib.load unpickles, which executes.
    for name in ("tfidf_word.pkl", "tfidf_char.pkl"):
        verify_model_hash(model_dir / name)

    hierarchy_path = data_dir / "hierarchy.json"
    bundle = {
        "tfidf_word": joblib.load(model_dir / "tfidf_word.pkl"),
        "tfidf_char": joblib.load(model_dir / "tfidf_char.pkl"),
        "hierarchy": json.loads(hierarchy_path.read_text()) if hierarchy_path.is_file() else {},
        "targets": {},
        "order": list(TARGETS),
    }

    for target, spec in TARGETS.items():
        raw = json.loads((data_dir / f"label_map_{target}.json").read_text())
        id2name = {int(k): v["name"] for k, v in raw.items()}
        target_dir = model_dir / target

        entry = {
            "id2name": id2name,
            "name2id": {v: k for k, v in id2name.items()},
            "parent": spec.get("parent"),
            "top_k": int(spec.get("top_k", 1)),
            "threshold": review_threshold(target),
        }

        if spec.get("parent"):
            routing_path = target_dir / "routing.json"
            if not routing_path.is_file():
                raise FileNotFoundError(f"models/{target}/routing.json")
            entry["routing"] = json.loads(routing_path.read_text())
            entry["per_parent"] = {
                int(child.name): _load_member_models(child)
                for child in sorted(target_dir.iterdir())
                if child.is_dir() and child.name.isdigit()
            }
        else:
            if not (target_dir / "svm.pkl").is_file():
                raise FileNotFoundError(f"models/{target}/svm.pkl")
            entry.update(_load_member_models(target_dir))

        bundle["targets"][target] = entry

    return bundle


# Predicts a child target inside whichever parent each row was assigned.
def _predict_child(
    entry: dict, parent_labels: Sequence[str], parent_name2id: Dict[str, int], X
) -> dict:
    """Return per-row label, confidence, source and ranked alternates.

    Every row is routed to the classifier trained for its parent, so the
    answer is always one of that parent's own children.  Rows whose parent has
    no model fall back to the child that parent most often takes, and are
    flagged, rather than being answered by a model that never saw them.
    """
    n_rows = len(parent_labels)
    num_labels = len(entry["id2name"])
    top_k = entry["top_k"]

    labels = [""] * n_rows
    confidence = np.zeros(n_rows, dtype=np.float32)
    sources = ["model"] * n_rows
    alternates: List[List[str]] = [[] for _ in range(n_rows)]

    routing = entry["routing"]
    deterministic = routing.get("deterministic", {})
    fallback = routing.get("fallback", {})

    by_parent: Dict[str, List[int]] = {}
    for row, parent in enumerate(parent_labels):
        by_parent.setdefault(str(parent), []).append(row)

    for parent, rows in by_parent.items():
        if not parent:
            for row in rows:
                sources[row] = "no parent"
            continue

        only_child = deterministic.get(parent)
        if only_child is not None:
            for row in rows:
                labels[row] = only_child
                confidence[row] = 1.0
                sources[row] = "only child of parent"
            continue

        parent_id = parent_name2id.get(parent)
        member = entry["per_parent"].get(parent_id) if parent_id is not None else None

        if not member or not member["models"]:
            answer = fallback.get(parent, "")
            for row in rows:
                labels[row] = answer
                confidence[row] = 0.0
                sources[row] = "no model for parent" if answer else "parent unknown"
            continue

        scores = _score(member, X[rows], num_labels)
        ids, ranked = top_k_from_scores(scores, top_k)
        for i, row in enumerate(rows):
            labels[row] = entry["id2name"].get(int(ids[i, 0]), "")
            confidence[row] = float(ranked[i, 0])
            alternates[row] = [
                entry["id2name"].get(int(c), "") for c in ids[i, 1:]
            ]

    return {
        "labels": labels,
        "confidence": confidence,
        "sources": sources,
        "alternates": alternates,
    }


# Classifies a list of cleaned texts against every configured target.
def classify(
    texts: Sequence[str],
    bundle: dict,
    overrides: Optional[Dict[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Return one row per text with a prediction column set per target.

    Targets are evaluated parent before child, so a child is always predicted
    inside the category its parent landed on.  `overrides` supplies a confirmed
    parent per row — a coder accepting or correcting the Process Category —
    which is worth 9.4 points of accuracy at the level below it.

    Args:
        texts: Cleaned input texts.
        bundle: The artifacts from `load_bundle`.
        overrides: target -> per-row label to use instead of the prediction.
            An empty string leaves that row to the model.

    Returns:
        A frame with pred_/confidence_/review_/source_ per target, plus
        alt_<target> where the target reports ranked alternatives.
    """
    overrides = overrides or {}
    texts = [str(t) for t in texts]
    out = pd.DataFrame(index=range(len(texts)))
    blank = [is_blank_text(t) for t in texts]

    X = build_features(texts, bundle["tfidf_word"], bundle["tfidf_char"])

    for target in bundle["order"]:
        entry = bundle["targets"][target]
        parent = entry["parent"]
        num_labels = len(entry["id2name"])

        if parent is None:
            scores = _score(entry, X, num_labels)
            ids, ranked = top_k_from_scores(scores, entry["top_k"])
            labels = [entry["id2name"].get(int(i), "") for i in ids[:, 0]]
            confidence = ranked[:, 0].astype(np.float32)
            sources = ["model"] * len(texts)
            alternates = [
                [entry["id2name"].get(int(c), "") for c in row] for row in ids[:, 1:]
            ]
        else:
            parent_entry = bundle["targets"][parent]
            parent_labels = out[f"pred_{parent}"].tolist()
            result = _predict_child(entry, parent_labels, parent_entry["name2id"], X)
            labels = result["labels"]
            confidence = result["confidence"]
            sources = result["sources"]
            alternates = result["alternates"]

        out[f"pred_{target}"] = labels
        out[f"confidence_{target}"] = confidence
        out[f"source_{target}"] = sources
        out[f"review_{target}"] = confidence < entry["threshold"]
        if entry["top_k"] > 1:
            out[f"alt_{target}"] = [", ".join(a for a in row if a) for row in alternates]

        # A coder's confirmed answer replaces the model's, and is never flagged.
        supplied = overrides.get(target)
        if supplied is not None:
            for row, value in enumerate(supplied):
                if not value:
                    continue
                out.at[row, f"pred_{target}"] = value
                out.at[row, f"confidence_{target}"] = 1.0
                out.at[row, f"review_{target}"] = False
                out.at[row, f"source_{target}"] = "confirmed by reviewer"

        # A row with no description gets no answer.  The models still score an
        # all-zero vector and can report high confidence doing it.
        for row, is_empty in enumerate(blank):
            if not is_empty:
                continue
            out.at[row, f"pred_{target}"] = ""
            out.at[row, f"confidence_{target}"] = 0.0
            out.at[row, f"review_{target}"] = True
            out.at[row, f"source_{target}"] = "no description"

        # A parent the model was unsure of makes its child unsure too.
        if parent is not None:
            inherited = out[f"review_{parent}"].to_numpy()
            confirmed = out[f"source_{target}"].to_numpy() == "confirmed by reviewer"
            out[f"review_{target}"] = (
                out[f"review_{target}"].to_numpy() | (inherited & ~confirmed)
            )

    return out
