#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.train
# Trains the TF-IDF vectorizers and one calibrated classifier per target,
# including a per-parent classifier for each hierarchical target.

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from src.config import (
    BRANCHES, CONFIG, CONFIG_PATH, DATA_DIR, FEEDBACK_PATH, GATE,
    HIERARCHY_PATH, MODEL_DIR, PROC_DIR, ROOT, SEED, TARGETS,
    branch_label, branch_title, label_map_path, target_title,
)
from src.data import clean_text
from src.feedback import feedback_training_rows
from src.models import expand_proba, hstack_csr, search_blend_weights, write_hashes
# Configures logging for the training pipeline.
logging.basicConfig(
    filename=ROOT / "train.log",
    level=logging.INFO,
    format="%(asctime)s [INFO] %(message)s",
)


# scikit-learn inspects a classifier's classes_ with a heuristic meant to catch
# a regression target passed to a classifier: it warns when more than half the
# values are unique and there are over 20 of them.  A classes_ array is unique
# by definition, so every target here with 21 or more categories trips it — at
# fit, once per calibration fold, and again at every predict.  Training the
# per-parent models for the deepest level emits it thousands of times and says
# nothing about the data either time, so it is filtered here rather than around
# each call.
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50%",
    category=UserWarning,
)


# Model families under the names a reader who is not a data scientist can
# follow.  The short keys are what the saved artifacts and the reports use.
MODEL_NAMES = {
    "svm": "Support vector machine",
    "sgd": "Stochastic gradient",
    "lr": "Logistic regression",
    "xgb": "Gradient boosting",
}


# Records a line in train.log without putting it on screen.
def log(msg):
    """Write to the training log only.

    Everything the run does is kept here.  The console shows the subset a
    person watching it needs, because a coding team reads this output to see
    whether the models are worth using, not to debug scikit-learn.
    """
    logging.info(msg)


# Puts a line on screen and in the log.
def say(msg=""):
    print(msg, flush=True)
    if msg.strip():
        logging.info(msg)


# Renders a macro-F1 as a score out of 100.
def score_of(value: float) -> str:
    """Format a macro-F1 for a reader who has never heard of one."""
    return "  n/a" if value != value else f"{value * 100:5.1f}"


# Prints a progress line and records it in the training log.
def info(msg):
    print(msg, flush=True)
    logging.info(msg)


# Vectorizers are shared by every target; only the classifiers differ, so
# these live at the root of models/ and each target gets a subfolder.
SHARED_ARTIFACTS = ["tfidf_word.pkl", "tfidf_char.pkl"]

# A parent needs at least this many training rows before its own child
# classifier is worth fitting; below it the majority child is recorded and the
# row is sent for review instead of being guessed at by a model built on
# almost nothing.
MIN_PARENT_ROWS = 10

# Whether logistic regression and boosting are trained alongside the linear SVM
# and kept where they beat it on validation macro-F1.
#
# They are given the whole training set and the boosting rounds the model was
# configured with, so the comparison answers "is this family better" rather
# than "is it better on a budget".  That is expensive — boosting builds one
# tree per category per round over a 180,000-column matrix, and the
# twenty-seven category target runs for hours — and it is worth it, because a
# cheap screen can only ever hint at the answer.
#
# Macro-F1 rather than accuracy decides it: accuracy on this data sits at the
# limit of how consistently people code it, so any gain a second model has to
# offer is in the rare categories, which only the macro average sees.
#
# Set max_rows above zero to screen on a sample instead; zero uses every row.
GATE_ENABLED = bool(GATE.get("enabled", True))
GATE_BOOST_ROUNDS = int(GATE.get("max_boost_rounds", 350))
GATE_EARLY_STOPPING = int(GATE.get("early_stopping_rounds", 50))
GATE_MAX_ROWS = int(GATE.get("max_rows", 0))
GATE_BOOST_FEATURES = int(GATE.get("boost_features", 30_000))

# Which families are put up against the linear SVM.  The hierarchical fields
# get the cheap two: their groups are small — some only a few dozen records —
# and boosting on a handful of rows measures noise rather than a method.
CHALLENGERS = tuple(GATE.get("challengers", ("sgd", "lr", "xgb")))
HIERARCHICAL_CHALLENGERS = tuple(GATE.get("hierarchical_challengers", ("sgd", "lr")))
HIERARCHICAL_MIN_ROWS = int(GATE.get("hierarchical_min_rows", 200))


# Removes old model artifacts before training.
def clean_old_models():
    """Empty models/ so a run never mixes fresh artifacts with stale ones.

    Deepest paths first, so a folder is only removed once emptied.  Failures
    are reported rather than swallowed: a file that cannot be replaced would
    otherwise leave a previous run's model in place, and inference would load
    it against this run's label map.
    """
    if not MODEL_DIR.exists():
        return

    failed = []
    for path in sorted(MODEL_DIR.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            path.unlink() if path.is_file() else path.rmdir()
        except OSError as exc:
            failed.append(f"{path.relative_to(MODEL_DIR)} ({exc.strerror})")

    if failed:
        raise RuntimeError(
            "Could not clear previous artifacts from models/: "
            + ", ".join(failed)
            + ". Remove them by hand, or stop whatever is holding them open."
        )


# Fits a calibrated linear SVM over the rows it is given.
def fit_svm(X, y, seed: int = SEED):
    """Return a calibrated LinearSVC, or None when the data cannot support one.

    Calibration is what makes `predict_proba` available, and the probability
    it yields is the number the review threshold and the coverage curve are
    both read against.  It needs at least as many members per class as folds,
    so classes thinner than that are dropped first; if fewer than two classes
    survive there is nothing to discriminate and None is returned.

    `class_weight="balanced"` is measured: on the Process Category split it
    gained 0.9 macro-F1 at no cost to accuracy, and the rare categories are
    where the model is weakest.
    """
    # pandas-stubs types value_counts() as ndarray; it returns a Series, and
    # the class labels are its index.
    counts = cast(pd.Series, pd.Series(y).value_counts())
    keep_classes = cast(pd.Series, counts[counts >= 2]).index
    if len(keep_classes) < 2:
        return None

    mask = pd.Series(y).isin(keep_classes).to_numpy()
    X, y = X[mask], np.asarray(y)[mask]

    folds = int(min(3, pd.Series(y).value_counts().min()))
    if folds < 2:
        return None

    model = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed),
        method="sigmoid",
        cv=folds,
    )

    model.fit(X, y)
    return model


# Produces one model's probability matrix over the full label space.
def proba_of(model, X, num_labels: int) -> np.ndarray:
    """Return (n_rows, num_labels) probabilities, whatever family `model` is.

    A Booster predicts from its own matrix type, over the reduced feature space
    it was fitted on, and already returns one column per class.
    """
    if isinstance(model, xgb.Booster):
        selector = getattr(model, "feature_selector", None)
        columns = selector.transform(X) if selector is not None else X
        return expand_proba(
            model.predict(xgb.QuantileDMatrix(columns)), np.arange(num_labels), num_labels
        )
    return expand_proba(model.predict_proba(X), model.classes_, num_labels)


# Scores a fitted model's macro-F1 on the validation rows.
def macro_f1(model, X, y) -> float:
    """Macro-F1 of `model` on (X, y), or 0.0 when there is nothing to score.

    Macro rather than plain accuracy because the categories are wildly
    unbalanced — one Process category holds 23% of rows and several hold under
    0.1% — and accuracy is almost unmoved by getting the whole tail wrong.
    """
    if model is None or len(y) == 0:
        return 0.0

    # A Booster predicts from its own matrix type and returns one probability
    # per class rather than a label, unlike every scikit-learn estimator here.
    if isinstance(model, xgb.Booster):
        selector = getattr(model, "feature_selector", None)
        columns = selector.transform(X) if selector is not None else X
        predicted = model.predict(xgb.DMatrix(columns)).argmax(axis=1)
    else:
        predicted = model.predict(X)

    # sklearn documents zero_division=0 as valid but annotates it as str.
    return float(
        f1_score(y, predicted, average="macro", zero_division=0)  # type: ignore[arg-type]
    )


# Decides which model families are worth keeping for a target.
def gate_models(
    target, X_train, y_train, X_val, y_val, num_labels,
    challengers: Optional[Sequence[str]] = None,
    quiet: bool = False,
) -> Tuple[dict, dict]:
    """Fit SVM, LR and XGBoost and keep only those that earn their place.

    The ensemble was previously a fixed 0.40/0.40/0.20 blend of all three.
    Measured on a held-out split, the calibrated linear SVM alone matched the
    full ensemble, so each of the other two now has to demonstrate a macro-F1
    gain on validation before it is carried.  Whatever survives is recorded in
    the target's ensemble.json alongside the scores that justified it, so the
    decision is auditable rather than asserted.

    Returns:
        (models, report) — the fitted models that passed, and what each scored.
    """
    report: Dict[str, float] = {}
    announce = (lambda msg: None) if quiet else say
    challengers = list(CHALLENGERS) if challengers is None else list(challengers)

    if not GATE_ENABLED or not challengers:
        svm_only = fit_svm(X_train, y_train)
        if svm_only is None:
            raise RuntimeError(f"{target}: not enough labelled data to fit a classifier.")
        score = macro_f1(svm_only, X_val, y_val)
        announce(f"            Method: {MODEL_NAMES['svm'].lower()} (others not compared)")
        announce(f"            Score: {score_of(score)}")
        return {"svm": svm_only}, {"svm": score}

    if GATE_MAX_ROWS and len(y_train) > GATE_MAX_ROWS:
        rng = np.random.default_rng(SEED)
        sample = rng.choice(len(y_train), GATE_MAX_ROWS, replace=False)
        Xg, yg = X_train[sample], np.asarray(y_train)[sample]
        log(f"{target}: screening challengers on {GATE_MAX_ROWS:,} of {len(y_train):,} rows")
    else:
        Xg, yg = X_train, np.asarray(y_train)

    svm_model = fit_svm(X_train, y_train)
    if svm_model is None:
        raise RuntimeError(f"{target}: not enough labelled data to fit a classifier.")

    baseline = macro_f1(svm_model, X_val, y_val)
    report["svm"] = baseline
    announce("            Comparing methods, keeping whichever scores best:")
    announce(f"              {MODEL_NAMES['svm']:.<34} {score_of(baseline)}")

    # Several unrelated estimator types share this dict; each is scored
    # through its own branch in save_models and src.inference.
    kept: Dict[str, Any] = {"svm": svm_model}

    # Each challenger is fitted inside `try_challenger` so one failing costs
    # its own result rather than the whole run, and so a score is reported the
    # moment it is known rather than at the decision below — an earlier run was
    # killed part way through and threw away two measurements it had already
    # finished paying for.
    #
    # This catches a Python-level failure only.  A process killed by the kernel
    # for exhausting its memory cgroup receives SIGKILL and can catch nothing,
    # which is why the boosting matrices below are built as QuantileDMatrix
    # rather than left to take whatever memory they ask for.
    def try_challenger(key: str, name: str, fit):
        log(f"{target}: fitting {name}")
        try:
            model = fit()
        except Exception as exc:
            announce(f"              {MODEL_NAMES[key]:.<34}   failed")
            log(f"{target}: {name} could not be fitted ({type(exc).__name__}: {exc})")
            report[key] = float("nan")
            return None
        report[key] = macro_f1(model, X_val, y_val)
        announce(f"              {MODEL_NAMES[key]:.<34} {score_of(report[key])}")
        return model

    # Each challenger is a name the caller can ask for, so the hierarchical
    # sweep can run the cheap two per parent and leave boosting to the flat
    # fields, where there is enough data for it to mean anything.
    def fit_sgd():
        # `modified_huber` rather than the default hinge: it is the loss that
        # gives SGDClassifier a predict_proba, and the blend needs probabilities
        # from every member or the weighted sum stops being one.  It also
        # reaches a different optimum than LinearSVC despite both being linear —
        # a smooth loss with early stopping against a hinge fitted to
        # convergence — which is the point of carrying it separately.
        return SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            early_stopping=True,
            n_iter_no_change=5,
            max_iter=2000,
            random_state=SEED,
        ).fit(Xg, yg)

    def fit_lr():
        return LogisticRegression(
            max_iter=2500, solver="lbfgs", C=1.5, class_weight="balanced",
        ).fit(Xg, yg)

    def fit_booster():
        # Boosting is the one family here that cannot be handed the full
        # feature space.  It builds a histogram per feature per class, so
        # 183,633 TF-IDF columns against even seven categories exhausted 13.7 GB
        # and the run was killed outright — twice, the second time after the
        # matrix itself had been made six times smaller, because the cost is in
        # training rather than in storing the data.
        #
        # It is given the most informative GATE_BOOST_FEATURES columns instead.
        # That is not a handicap so much as what trees are for: a tree splits on
        # individual features and cannot use a hundred thousand of them, while a
        # linear model weights all of them at once.
        selector = SelectKBest(chi2, k=min(GATE_BOOST_FEATURES, Xg.shape[1]))
        Xb = selector.fit_transform(Xg, yg)
        Xb_val = selector.transform(X_val)
        log(f"{target}: boosting on {Xb.shape[1]:,} of {Xg.shape[1]:,} features")

        params = {
            "objective": "multi:softprob",
            "num_class": num_labels,
            "max_depth": 6,
            "learning_rate": 0.12,
            "eval_metric": "mlogloss",
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            # Fewer histogram bins per feature, which is the other half of the
            # memory this uses; 64 is ample for TF-IDF values.
            "max_bin": 64,
            "reg_lambda": 1.0,
            "random_state": SEED,
        }
        # max_bin has to be given to the matrix as well as to the booster;
        # XGBoost refuses to train when the two disagree.
        train_matrix = xgb.QuantileDMatrix(Xb, label=yg, max_bin=params["max_bin"])
        val_matrix = xgb.QuantileDMatrix(
            Xb_val, label=y_val, ref=train_matrix, max_bin=params["max_bin"]
        )
        booster = xgb.train(
            params,
            train_matrix,
            num_boost_round=GATE_BOOST_ROUNDS,
            evals=[(val_matrix, "val")],
            early_stopping_rounds=GATE_EARLY_STOPPING,
            verbose_eval=False,
        )
        # The selector travels with the booster: whatever scores or predicts
        # with it later must see the same columns it was fitted on.
        booster.feature_selector = selector  # type: ignore[attr-defined]
        return booster

    fits = {"sgd": ("SGD (modified huber)", fit_sgd),
            "lr": ("Logistic Regression", fit_lr),
            "xgb": ("XGBoost", fit_booster)}

    for key in challengers:
        if key not in fits:
            raise ValueError(f"Unknown challenger '{key}'; known: {', '.join(fits)}")
        name, fit = fits[key]
        model = try_challenger(key, name, fit)
        if model is None:
            continue
        if report[key] > baseline:
            kept[key] = model
            log(f"{target}: keeping {key} ({report[key]:.4f} > {baseline:.4f})")
        else:
            log(f"{target}: dropping {key} ({report[key]:.4f} <= {baseline:.4f})")

    if len(kept) == 1:
        announce(f"            Using:  {MODEL_NAMES['svm'].lower()} — the others scored lower")
    else:
        blended = ", ".join(MODEL_NAMES[n].lower() for n in kept)
        announce(f"            Using:  {blended}, blended together")

    return kept, report


# Decides how much of the blend each surviving model should carry.
def weigh_blend(models: dict, X_val, y_val, num_labels: int, announce) -> Tuple[dict, dict]:
    """Search the blend weights on validation, and report what it bought.

    Splitting evenly is the obvious choice and the wrong one: two models that
    both cleared the gate are not therefore equally good.  A weighting is only
    adopted where it beats the even split on the same measure the gate used.
    """
    if len(models) < 2:
        return {name: 1.0 for name in models}, {}

    probas = {name: proba_of(model, X_val, num_labels) for name, model in models.items()}
    weights, best, even = search_blend_weights(probas, np.asarray(y_val), num_labels)

    detail = {"searched": round(best, 6), "even_split": round(even, 6)}
    if best > even:
        shown = ", ".join(f"{MODEL_NAMES[n].lower()} {w:.0%}" for n, w in weights.items() if w)
        announce(f"            Blend:  {shown}  (+{(best - even) * 100:.1f} over an even split)")
    else:
        announce("            Blend:  even split — no weighting beat it")
    return weights, detail


# Saves a target's models and records the weighting inference should use.
def save_models(
    out_dir: Path, models: dict, report: dict,
    weights: Optional[Dict[str, float]] = None,
    blend: Optional[dict] = None,
) -> None:
    """Write each kept model plus the ensemble.json describing the blend."""
    weights = weights or {name: round(1.0 / len(models), 4) for name in models}

    # The search can weigh a member to zero, which is the honest answer that it
    # earns no place in the blend.  Writing it anyway would ship a model that
    # is loaded, verified and then multiplied by nothing.
    weights = {name: weight for name, weight in weights.items() if weight > 0}
    models = {name: model for name, model in models.items() if name in weights}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for name, model in models.items():
        if name == "xgb":
            path = out_dir / "xgb.json"
            model.save_model(str(path))
        else:
            path = out_dir / f"{name}.pkl"
            joblib.dump(model, path)
        written.append(path)

    payload = {"weights": weights, "validation_macro_f1": report}
    if blend is not None:
        payload["blend_search"] = blend
    (out_dir / "ensemble.json").write_text(json.dumps(payload, indent=4))
    write_hashes(written)


# Trains the classifier that decides whether a family of codes applies at all.
def train_branches(train_df, val_df, X_train, X_val) -> None:
    """Fit one yes/no classifier per branch and save it beside the targets.

    The in-branch models are trained only on records that carry the branch,
    because a blank is an absence rather than a category — that is what stops
    "uncoded" becoming the largest class.  The consequence is that they have
    never seen a record the branch does not apply to and will answer one just
    as confidently as any other.  Program codes appear on 72.9% of SURV
    records and 6.1% of TIRs, so without this the models would put a Program
    code on almost every TIR.

    Every record trains this, coded or not: here an empty column is not a
    missing label, it is the negative answer.
    """
    if not BRANCHES:
        return

    say()
    say("Step 3 of 4   Learning which families of codes apply")

    for branch in BRANCHES:
        column = branch_label(branch)
        if column not in train_df.columns:
            raise ValueError(
                f"{column} is not in the training split. Re-run preprocessing "
                f"after adding the '{branch}' branch to {CONFIG_PATH.name}."
            )

        y_tr = train_df[column].astype(int).to_numpy()
        y_va = val_df[column].astype(int).to_numpy()

        say()
        say(f"  {branch_title(branch)}")
        say(f"            applies to {int(y_tr.sum()):,} of {len(y_tr):,} "
            f"TIRs learned from ({y_tr.mean():.0%})")

        models, report = gate_models(
            f"branch_{branch}", X_train, y_tr, X_val, y_va, 2, quiet=True,
        )
        weights, blend = weigh_blend(models, X_val, y_va, 2, log)
        save_models(MODEL_DIR / f"branch_{branch}", models, report, weights, blend)

        best = blend.get("searched") or max(report.get(n, 0.0) for n in models)
        say(f"            score {score_of(best)}")


# Trains one classifier per parent category for a hierarchical target.
def train_hierarchical(
    target: str, spec: dict, label_maps: dict, hierarchy: dict,
    train_df, val_df, X_train, X_val,
) -> float:
    """Fit a separate classifier for each parent, over that parent's children.

    Predicting 1,079 third-level categories with one model is not workable:
    the coefficient matrix alone runs to hundreds of megabytes, and boosting
    would build a tree per class per round.  Splitting by parent turns it into
    roughly 170 classifiers over five to fifteen classes each, which is both
    tractable and, by construction, incapable of returning a child that does
    not belong to its parent.

    Each model is fitted on the *true* parent label.  At prediction time the
    parent is whatever was predicted (or confirmed by a coder), so a wrong
    parent still costs the child — measured at 9.4 points on the second level,
    which is why the UI lets a coder fix the parent first.
    """
    parent = spec["parent"]
    out_root = MODEL_DIR / target
    out_root.mkdir(parents=True, exist_ok=True)

    child_names = {int(k): v["name"] for k, v in label_maps[target].items()}
    parent_names = {int(k): v["name"] for k, v in label_maps[parent].items()}
    name2parent_id = {v: k for k, v in parent_names.items()}
    num_labels = len(child_names)

    tr_parent = train_df[f"label_{parent}"].to_numpy()
    tr_child = train_df[f"label_{target}"].to_numpy()
    va_parent = val_df[f"label_{parent}"].to_numpy()
    va_child = val_df[f"label_{target}"].to_numpy()

    fallbacks: Dict[str, str] = {}
    deterministic: Dict[str, str] = {}
    blended: Dict[str, List[str]] = {}
    fitted = skipped = 0
    scores: List[float] = []

    for parent_name in sorted(hierarchy.get(target, {})):
        parent_id = name2parent_id.get(parent_name)
        if parent_id is None:
            continue

        rows = (tr_parent == parent_id) & (tr_child >= 0)
        n_rows = int(rows.sum())
        if n_rows == 0:
            continue

        children = np.unique(tr_child[rows])
        majority = child_names[int(pd.Series(tr_child[rows]).mode().iloc[0])]

        if len(children) == 1:
            deterministic[parent_name] = child_names[int(children[0])]
            continue

        if n_rows < MIN_PARENT_ROWS:
            fallbacks[parent_name] = majority
            skipped += 1
            continue

        val_rows = (va_parent == parent_id) & (va_child >= 0)

        # A parent only gets its challengers screened when it has enough of its
        # own records for the comparison to mean anything.  Below that the
        # groups are a few dozen rows and the ranking between families is
        # noise, so the SVM ships without a contest.
        big_enough = n_rows >= HIERARCHICAL_MIN_ROWS and val_rows.sum() >= 20
        challengers = list(HIERARCHICAL_CHALLENGERS) if big_enough else []

        try:
            models, report = gate_models(
                target,
                X_train[rows], tr_child[rows],
                X_val[val_rows], va_child[val_rows],
                num_labels,
                challengers=challengers,
                quiet=True,
            )
        except RuntimeError:
            fallbacks[parent_name] = majority
            skipped += 1
            continue

        weights, _ = weigh_blend(
            models, X_val[val_rows], va_child[val_rows], num_labels, lambda msg: None
        )
        save_models(out_root / str(parent_id), models, report, weights)

        if len(models) > 1:
            blended[parent_name] = sorted(models)
        fallbacks.setdefault(parent_name, majority)
        fitted += 1

        if val_rows.sum():
            best = models[max(models, key=lambda n: report.get(n, 0.0))]
            scores.append(macro_f1(best, X_val[val_rows], va_child[val_rows]))

    (out_root / "routing.json").write_text(json.dumps({
        "parent": parent,
        "deterministic": deterministic,
        "fallback": fallbacks,
        "num_labels": num_labels,
    }, indent=4))

    mean_score = float(np.mean(scores)) if scores else 0.0
    parent_title = target_title(parent)
    total_groups = fitted + len(deterministic) + skipped
    say(f"            Chosen inside {parent_title}, so it can only return a code")
    say(f"            that belongs under the {parent_title} it was given.")
    say(f"            {total_groups} groups of codes:")
    say(f"              {fitted} learned from examples")
    if deterministic:
        say(f"              {len(deterministic)} had only one possible answer")
    if skipped:
        say(f"              {skipped} had too few examples, so a coder is asked instead")
    say(f"            Score: {score_of(mean_score)}")
    log(
        f"{target}: {fitted} per-parent models, {len(deterministic)} deterministic, "
        f"{skipped} too thin; mean per-parent validation macro-F1 {mean_score:.4f}"
    )
    return mean_score


# Runs the complete training pipeline.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-gate", action="store_true",
        help="Ship the calibrated SVM alone without training the challengers. "
             "Otherwise all three are fitted and the best-scoring are kept.",
    )
    args = parser.parse_args()

    global GATE_ENABLED
    if args.no_gate:
        GATE_ENABLED = False

    started = time.time()
    say("=" * 70)
    say("  Training the TIR liability coding models")
    say("=" * 70)
    log("Starting training")

    if not TARGETS:
        raise ValueError(f"{CONFIG_PATH.name} is missing 'targets'.")

    clean_old_models()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # low_memory=False reads each column in one pass.  Chunked reading makes
    # pandas guess a dtype per chunk and warn about the ones that disagree —
    # which they do, because the split carries through every column the
    # original exports had, flags and free text together.
    train_df = pd.read_csv(PROC_DIR / "train.csv", low_memory=False)
    val_df = pd.read_csv(PROC_DIR / "val.csv", low_memory=False)

    for df in (train_df, val_df):
        df["text"] = df["text"].astype(str).apply(clean_text)

    say()
    say("Step 1 of 4   Reading the coded TIRs")
    say(f"              {len(train_df):,} to learn from, "
        f"{len(val_df):,} held back to mark the work against")

    label_maps = {
        t: json.loads(label_map_path(t).read_text()) for t in TARGETS
    }
    hierarchy = json.loads(HIERARCHY_PATH.read_text()) if HIERARCHY_PATH.is_file() else {}

    # Folds reviewer corrections from the web app into the training split, so
    # the models themselves learn from the mistakes.  Corrections only ever
    # join training data: validation and test stay untouched so metrics remain
    # honest.  Each correction names the target it applies to.
    for target, label_map in label_maps.items():
        name2id = {v["name"]: int(k) for k, v in label_map.items()}
        corrections = feedback_training_rows(FEEDBACK_PATH, name2id, target=target)

        if corrections.empty:
            log(f"No reviewer corrections to fold into {target}.")
            continue

        extra = pd.DataFrame({
            "text": corrections["text"],
            **{f"label_{t}": (corrections["label"] if t == target else -1) for t in TARGETS},
        })
        train_df = pd.concat([train_df, extra], ignore_index=True)
        say(f"              {len(extra)} correction(s) you saved were folded into "
            f"{target_title(target)}")
        log(f"Added {len(extra)} reviewer corrections to {target}")

    # Trains TF-IDF vectorizers on training data.  These are unsupervised, so
    # one pair serves every target.
    say()
    say("Step 2 of 4   Learning the vocabulary of the descriptions")

    tfidf_word = TfidfVectorizer(
        lowercase=True,
        strip_accents="ascii",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        max_features=200_000,
        # float32 halves the matrix's memory over the default float64 and
        # costs no measurable accuracy.  sklearn accepts it but annotates
        # the parameter as float64 only, hence the ignore.
        dtype=np.float32,  # type: ignore[arg-type]
        sublinear_tf=True,
    )
    tfidf_char = TfidfVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.95,
        max_features=100_000,
        dtype=np.float32,  # type: ignore[arg-type]
        sublinear_tf=True,
    )

    X_train = hstack_csr([
        tfidf_word.fit_transform(train_df["text"]),
        tfidf_char.fit_transform(train_df["text"]),
    ])
    X_val = hstack_csr([
        tfidf_word.transform(val_df["text"]),
        tfidf_char.transform(val_df["text"]),
    ])
    # scipy annotates a sparse matrix's shape as optional; hstack_csr always
    # returns a concrete two-dimensional CSR matrix.
    rows, columns = cast(tuple, X_train.shape)
    say(f"              {columns:,} words and word-fragments found across {rows:,} TIRs")
    log(f"feature matrix: {rows:,} x {columns:,}")

    joblib.dump(tfidf_word, MODEL_DIR / "tfidf_word.pkl")
    joblib.dump(tfidf_char, MODEL_DIR / "tfidf_char.pkl")
    write_hashes([MODEL_DIR / name for name in SHARED_ARTIFACTS])

    # Learns which families of codes each record carries before learning the
    # codes themselves, since a branch that does not apply is not a field to
    # be filled in wrongly.
    train_branches(train_df, val_df, X_train, X_val)

    # Trains one set of classifiers per target, on the rows that target can
    # use.  A row missing this label carries -1 and is skipped here, while
    # still contributing to any other target it does have.
    say()
    say("Step 4 of 4   Learning to code each field")
    summary = []

    for position, (target, spec) in enumerate(TARGETS.items(), start=1):
        col = f"label_{target}"
        tr_rows = train_df[col].to_numpy() >= 0
        va_rows = val_df[col].to_numpy() >= 0
        num_labels = len(label_maps[target])

        say()
        say(f"  [{position} of {len(TARGETS)}]  {target_title(target)}")
        say(f"            {num_labels} categories, learned from "
            f"{int(tr_rows.sum()):,} coded TIRs")
        log(f"Training {target}: {int(tr_rows.sum()):,} rows, {num_labels} categories")

        if spec.get("parent"):
            score = train_hierarchical(
                target, spec, label_maps, hierarchy,
                train_df, val_df, X_train, X_val,
            )
        else:
            y_tr = train_df.loc[tr_rows, col].astype(int).to_numpy()
            y_va = val_df.loc[va_rows, col].astype(int).to_numpy()
            models, report = gate_models(
                target, X_train[tr_rows], y_tr, X_val[va_rows], y_va, num_labels,
            )
            weights, blend = weigh_blend(models, X_val[va_rows], y_va, num_labels, say)
            save_models(MODEL_DIR / target, models, report, weights, blend)
            score = blend.get("searched") or max(report.get(n, 0.0) for n in models)

        summary.append((target_title(target), num_labels, score))

    minutes = (time.time() - started) / 60
    say()
    say("=" * 70)
    say(f"  Finished in {minutes:.0f} minute(s)")
    say("=" * 70)
    say()
    for title, categories, score in summary:
        say(f"  {title:<16}{categories:>5} categories    score {score_of(score)}")
    say()
    say("  The score is out of 100 and counts every category equally, so a")
    say("  category the team codes daily and one they code twice a year weigh")
    say("  the same. It is deliberately harsher than \"percent correct\" —")
    say("  reports/benchmark.md gives that, and what it means for coder time.")
    say()
    say("  The models are saved. To use them:")
    say("      streamlit run src/app.py")
    say()
    log("Training complete.")


if __name__ == "__main__":
    main()
