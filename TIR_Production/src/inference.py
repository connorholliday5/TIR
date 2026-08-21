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
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence
import joblib
import numpy as np
import pandas as pd

from src.config import (
    BRANCHES, CONFIG, DATA_DIR, LLM, MODEL_DIR, PRIMARY_TARGET, TARGETS,
    branch_threshold, branch_title, review_threshold, target_title,
)
from src.data import is_blank_text
from src.llm import CodexBackend, FakeBackend, load_definitions, suggest
from src.models import (
    build_features, ensemble_score_matrix, expand_proba,
    top_k_from_scores, verify_model_hash,
)


# The vectorizers every target shares.
SHARED_ARTIFACTS = ("tfidf_word.pkl", "tfidf_char.pkl")


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
    _require_artifacts(model_dir, data_dir)

    # Integrity first: joblib.load unpickles, which executes.
    for name in SHARED_ARTIFACTS:
        verify_model_hash(model_dir / name)

    hierarchy_path = data_dir / "hierarchy.json"
    return {
        "tfidf_word": joblib.load(model_dir / "tfidf_word.pkl"),
        "tfidf_char": joblib.load(model_dir / "tfidf_char.pkl"),
        "hierarchy": json.loads(hierarchy_path.read_text()) if hierarchy_path.is_file() else {},
        "targets": {
            target: _load_target(target, spec, model_dir, data_dir)
            for target, spec in TARGETS.items()
        },
        "branches": {
            branch: _load_branch(branch, model_dir)
            for branch in BRANCHES
        },
        "order": list(TARGETS),
    }


# Loads the classifier deciding whether one family of codes applies.
def _load_branch(branch: str, model_dir: Path) -> dict:
    """Return the models and threshold for one branch."""
    branch_dir = model_dir / f"branch_{branch}"
    if not (branch_dir / "svm.pkl").is_file():
        raise FileNotFoundError(f"models/branch_{branch}/svm.pkl")

    entry = {"threshold": branch_threshold(branch), "title": branch_title(branch)}
    entry.update(_load_member_models(branch_dir))
    return entry


# Checks every artifact exists before any of them is opened.
def _require_artifacts(model_dir: Path, data_dir: Path) -> None:
    """Raise naming everything that is absent, rather than the first one.

    A part-built model directory is the normal state before training has run,
    and a reader is better served by the whole list than by discovering it one
    file at a time.
    """
    missing = [n for n in SHARED_ARTIFACTS if not (model_dir / n).is_file()]
    missing += [
        f"data/label_map_{target}.json"
        for target in TARGETS
        if not (data_dir / f"label_map_{target}.json").is_file()
    ]
    missing += [
        f"models/branch_{branch}/svm.pkl"
        for branch in BRANCHES
        if not (model_dir / f"branch_{branch}" / "svm.pkl").is_file()
    ]
    if missing:
        raise FileNotFoundError(", ".join(missing))


# Loads the label map and classifiers for one target.
def _load_target(target: str, spec: dict, model_dir: Path, data_dir: Path) -> dict:
    """Return everything `classify` needs to answer for one field."""
    raw = json.loads((data_dir / f"label_map_{target}.json").read_text())
    id2name = {int(k): v["name"] for k, v in raw.items()}

    entry = {
        "id2name": id2name,
        "name2id": {v: k for k, v in id2name.items()},
        "parent": spec.get("parent"),
        "branch": spec.get("branch"),
        "top_k": int(spec.get("top_k", 1)),
        "threshold": review_threshold(target),
    }

    target_dir = model_dir / target
    if spec.get("parent"):
        entry.update(_load_per_parent(target, target_dir))
    else:
        if not (target_dir / "svm.pkl").is_file():
            raise FileNotFoundError(f"models/{target}/svm.pkl")
        entry.update(_load_member_models(target_dir))

    return entry


# Loads the routing table and one classifier per parent category.
def _load_per_parent(target: str, target_dir: Path) -> dict:
    """Return the routing table and the per-parent models beneath it."""
    routing_path = target_dir / "routing.json"
    if not routing_path.is_file():
        raise FileNotFoundError(f"models/{target}/routing.json")

    return {
        "routing": json.loads(routing_path.read_text()),
        "per_parent": {
            int(child.name): _load_member_models(child)
            for child in sorted(target_dir.iterdir())
            if child.is_dir() and child.name.isdigit()
        },
    }


# One target's answer for every row, before the rules below are applied.
class _Prediction(NamedTuple):
    labels: List[str]
    confidence: np.ndarray
    sources: List[str]
    alternates: List[List[str]]


# An unanswered prediction for `n_rows`, filled in by whichever path applies.
def _blank_prediction(n_rows: int) -> _Prediction:
    return _Prediction(
        labels=[""] * n_rows,
        confidence=np.zeros(n_rows, dtype=np.float32),
        sources=["model"] * n_rows,
        alternates=[[] for _ in range(n_rows)],
    )


# Gives a whole group of rows the same answer.
def _mark(
    prediction: _Prediction, rows: Sequence[int],
    label: str = "", confidence: float = 0.0, source: str = "model",
) -> None:
    """Write one answer across `rows` — used where no model is consulted."""
    for row in rows:
        prediction.labels[row] = label
        prediction.confidence[row] = confidence
        prediction.sources[row] = source


# Scores rows against a model and records the ranked result.
def _score_rows(
    prediction: _Prediction, entry: dict, member: dict, rows: Sequence[int], X
) -> None:
    """Predict `rows` with `member` and write the best label plus alternates."""
    num_labels = len(entry["id2name"])
    ids, ranked = top_k_from_scores(_score(member, X[rows], num_labels), entry["top_k"])

    for position, row in enumerate(rows):
        prediction.labels[row] = entry["id2name"].get(int(ids[position, 0]), "")
        prediction.confidence[row] = float(ranked[position, 0])
        prediction.alternates[row] = [
            entry["id2name"].get(int(c), "") for c in ids[position, 1:]
        ]


# Groups row numbers by the parent category each was assigned.
def _rows_by_parent(parent_labels: Sequence[str]) -> Dict[str, List[int]]:
    """Return {parent label: row numbers}, so each group is scored once."""
    grouped: Dict[str, List[int]] = {}
    for row, parent in enumerate(parent_labels):
        grouped.setdefault(str(parent), []).append(row)
    return grouped


# Finds the classifier belonging to one parent, if it has one.
def _model_for_parent(entry: dict, parent: str, parent_name2id: Mapping[str, int]):
    """Return the per-parent member models, or None if that parent has none."""
    parent_id = parent_name2id.get(parent)
    member = entry["per_parent"].get(parent_id) if parent_id is not None else None
    return member if member and member["models"] else None


# Predicts a flat target — one with no parent above it.
def _predict_flat(entry: dict, X, n_rows: int) -> _Prediction:
    """Score every row against this target's own models."""
    prediction = _blank_prediction(n_rows)
    _score_rows(prediction, entry, entry, range(n_rows), X)
    return prediction


# Predicts a child target inside whichever parent each row was assigned.
def _predict_child(
    entry: dict, parent_labels: Sequence[str], parent_name2id: Mapping[str, int], X
) -> _Prediction:
    """Route each row to the classifier trained for its parent.

    The answer is therefore always one of that parent's own children, and a
    combination the taxonomy does not contain cannot be produced.  Four cases
    are possible and each is handled explicitly — a row falling through them
    silently would be a confidently wrong answer with nothing raised.
    """
    prediction = _blank_prediction(len(parent_labels))
    routing = entry["routing"]
    deterministic = routing.get("deterministic", {})
    fallback = routing.get("fallback", {})

    for parent, rows in _rows_by_parent(parent_labels).items():
        if not parent:
            # The level above was withheld, so there is nothing to choose within.
            _mark(prediction, rows, source="no parent")
            continue

        only_child = deterministic.get(parent)
        if only_child is not None:
            _mark(prediction, rows, only_child, 1.0, "only child of parent")
            continue

        member = _model_for_parent(entry, parent, parent_name2id)
        if member is None:
            # Too few records under this parent to have modelled it; answer
            # with the child it most often takes and send the row for review.
            answer = fallback.get(parent, "")
            source = "no model for parent" if answer else "parent unknown"
            _mark(prediction, rows, answer, 0.0, source)
            continue

        _score_rows(prediction, entry, member, rows, X)

    return prediction


# Decides, per row, whether a family of codes applies to the record at all.
def _predict_branches(bundle: dict, X) -> Dict[str, np.ndarray]:
    """Return {branch: bool per row} — True where the branch's codes apply.

    A record below the branch's threshold is treated as not carrying the
    branch.  That is deliberately the cautious direction: leaving a field for a
    coder costs a moment, whereas filling in a family of codes the record
    should not carry is a wrong answer that looks like a confident one.
    """
    return {
        branch: _score(entry, X, 2)[:, 1] >= entry["threshold"]
        for branch, entry in bundle.get("branches", {}).items()
    }


# Withholds a target on the rows whose branch does not apply.
def _withhold_branch(
    out: pd.DataFrame, target: str, applies: np.ndarray, title: str
) -> None:
    """Blank `target` wherever its branch was judged not to apply.

    Written after the prediction rather than instead of it so the model still
    runs once for every row — the branch decides what is shown, and keeping
    the two separable is what lets the benchmark report the cost of the branch
    being wrong.
    """
    rows = ~applies
    if not rows.any():
        return

    out.loc[rows, f"pred_{target}"] = ""
    out.loc[rows, f"confidence_{target}"] = 0.0
    out.loc[rows, f"review_{target}"] = False
    out.loc[rows, f"source_{target}"] = f"no {title.lower()} on this record"
    if f"alt_{target}" in out.columns:
        out.loc[rows, f"alt_{target}"] = ""


# The codes available to a row, given the parent it was routed into.
def _candidates_for(
    bundle: dict, target: str, entry: dict, parent_label: str
) -> List[str]:
    """Return the codes a row could legitimately be given.

    For a child target that is its parent's observed children, so a suggestion
    cannot produce a combination QPS would reject; for a flat target it is the
    whole taxonomy.
    """
    parent = entry["parent"]
    if not parent:
        return list(entry["id2name"].values())

    children = bundle.get("hierarchy", {}).get(target, {}).get(parent_label, [])
    return list(children)


# Asks the language model about the rows the classifiers could not code.
def _consult_llm(
    out: pd.DataFrame, target: str, entry: dict, bundle: dict,
    texts: Sequence[str], backend, definitions: Mapping[str, Mapping[str, str]],
) -> None:
    """Fill in rows flagged for review, where the model names a valid code.

    Only rows the classifiers were not confident about reach here.  That is the
    whole design: the classifiers already match how consistently people code
    these fields, so there is nothing for a language model to win on the rows
    they are sure of, and asking about them would cost a call each for 40,000
    records a year.  What is left is the rare tail, where a definition can
    reach a code that has two examples behind it.

    A row the model declines, or answers outside its candidate list, keeps the
    classifier's answer and stays flagged.
    """
    field = target_title(target)
    limit = int(LLM.get("max_candidates", 40))
    parents = out[f"pred_{entry['parent']}"].tolist() if entry["parent"] else [""] * len(texts)

    # Only present when a model was actually consulted, so a run without one
    # produces exactly the frame it did before.
    rationale = f"rationale_{target}"
    if rationale not in out.columns:
        out[rationale] = ""

    for row in np.flatnonzero(out[f"review_{target}"].to_numpy()):
        if is_blank_text(texts[row]):
            continue

        candidates = _candidates_for(bundle, target, entry, parents[row])
        if not candidates or len(candidates) > limit:
            continue

        code, reason = suggest(
            texts[row], field, candidates, backend, definitions.get(target, {})
        )
        if not code:
            continue

        out.loc[row, f"pred_{target}"] = code
        out.loc[row, f"source_{target}"] = "language model"
        out.loc[row, rationale] = reason


# Writes one target's prediction into the output frame.
def _write_prediction(
    out: pd.DataFrame, target: str, entry: dict, prediction: _Prediction
) -> None:
    out[f"pred_{target}"] = prediction.labels
    out[f"confidence_{target}"] = prediction.confidence
    out[f"source_{target}"] = prediction.sources
    out[f"review_{target}"] = prediction.confidence < entry["threshold"]
    if entry["top_k"] > 1:
        out[f"alt_{target}"] = [
            ", ".join(a for a in row if a) for row in prediction.alternates
        ]


# Replaces predictions a coder has confirmed.
def _apply_overrides(out: pd.DataFrame, target: str, supplied: Sequence[str]) -> None:
    """A confirmed answer replaces the model's and is never flagged."""
    for row, value in enumerate(supplied):
        if not value:
            continue
        out.at[row, f"pred_{target}"] = value
        out.at[row, f"confidence_{target}"] = 1.0
        out.at[row, f"review_{target}"] = False
        out.at[row, f"source_{target}"] = "confirmed by reviewer"


# Withholds any answer for rows with nothing to read.
def _withhold_blank(out: pd.DataFrame, target: str, blank: Sequence[bool]) -> None:
    """Blank the answer where there was no description.

    The models score an all-zero feature vector rather than refusing, and can
    report high confidence doing it — an empty string once scored 99.8%.
    """
    for row, is_empty in enumerate(blank):
        if not is_empty:
            continue
        out.at[row, f"pred_{target}"] = ""
        out.at[row, f"confidence_{target}"] = 0.0
        out.at[row, f"review_{target}"] = True
        out.at[row, f"source_{target}"] = "no description"


# Passes a parent's uncertainty down to its child.
def _inherit_parent_review(out: pd.DataFrame, target: str, parent: str) -> None:
    """Flag a child whose parent was flagged, unless a person confirmed it.

    A wrong parent guarantees a wrong child — measured at 9.4 points on the
    level below — so the doubt has to travel with it.
    """
    inherited = out[f"review_{parent}"].to_numpy()
    confirmed = out[f"source_{target}"].to_numpy() == "confirmed by reviewer"
    out[f"review_{target}"] = out[f"review_{target}"].to_numpy() | (inherited & ~confirmed)


# Classifies a list of cleaned texts against every configured target.
def classify(
    texts: Sequence[str],
    bundle: dict,
    overrides: Optional[Mapping[str, Sequence[str]]] = None,
    backend=None,
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
        backend: language model consulted for the rows the classifiers flag for
            review. None uses whatever config.json's `llm` block enables, which
            is nothing by default.

    Returns:
        A frame with pred_/confidence_/review_/source_ per target, plus
        alt_<target> where the target reports ranked alternatives, and
        rationale_<target> where a language model supplied the answer.
    """
    overrides = overrides or {}
    texts = [str(t) for t in texts]
    blank = [is_blank_text(t) for t in texts]
    out = pd.DataFrame(index=range(len(texts)))

    backend = backend if backend is not None else _configured_backend()
    definitions = load_definitions() if backend is not None else {}

    X = build_features(texts, bundle["tfidf_word"], bundle["tfidf_char"])

    # Which families of codes each record carries, decided once for all the
    # targets belonging to a branch so they cannot disagree with each other.
    # Written out as its own column: the UI greys a family it does not apply
    # to rather than showing empty fields, and the benchmark cannot report what
    # the branch costs when it is wrong unless it can see the decision.
    applies = _predict_branches(bundle, X)
    for branch, verdict in applies.items():
        out[f"branch_{branch}"] = verdict

    for target in bundle["order"]:
        entry = bundle["targets"][target]
        parent = entry["parent"]

        if parent is None:
            prediction = _predict_flat(entry, X, len(texts))
        else:
            prediction = _predict_child(
                entry,
                out[f"pred_{parent}"].tolist(),
                bundle["targets"][parent]["name2id"],
                X,
            )

        # Order matters below, each rule outranking the one before it: the
        # branch clears fields the record does not carry, a coder's answer
        # replaces both the model's and the branch's judgement, a row with
        # nothing to read overrides even that, and the inherited doubt is
        # applied last so it can see which rows survived.
        _write_prediction(out, target, entry, prediction)

        branch = entry["branch"]
        if branch and branch in applies:
            _withhold_branch(out, target, applies[branch], bundle["branches"][branch]["title"])

        supplied = overrides.get(target)
        if supplied is not None:
            _apply_overrides(out, target, supplied)

        _withhold_blank(out, target, blank)

        if parent is not None:
            _inherit_parent_review(out, target, parent)

        # Last, so it sees the final set of rows still wanting an answer.
        if backend is not None:
            _consult_llm(out, target, entry, bundle, texts, backend, definitions)

    return out


# The language model config.json asks for, if it asks for one.
def _configured_backend():
    """Return the backend named by the `llm` block, or None when disabled.

    Off unless switched on, and the real backend additionally refuses to send
    anything until the coding team has sign-off — two separate switches,
    because "wire it up" and "send it records" are different decisions.
    """
    if not LLM.get("enabled", False):
        return None

    if LLM.get("backend") == "codex":
        return CodexBackend()
    return FakeBackend()
