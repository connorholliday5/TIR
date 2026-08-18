#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.train
# Trains the TF-IDF vectorizers and one calibrated classifier per target,
# including a per-parent classifier for each hierarchical target.

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

try:  # modules under src/
    from src.config import (
        CONFIG, CONFIG_PATH, DATA_DIR, FEEDBACK_PATH, GATE, HIERARCHY_PATH,
        MODEL_DIR, PROC_DIR, ROOT, SEED, TARGETS, label_map_path,
    )
    from src.data import clean_text
    from src.feedback import feedback_training_rows
    from src.models import hstack_csr, write_hashes
except ImportError:  # flat layout, modules at the repository root
    from config import (
        CONFIG, CONFIG_PATH, DATA_DIR, FEEDBACK_PATH, GATE, HIERARCHY_PATH,
        MODEL_DIR, PROC_DIR, ROOT, SEED, TARGETS, label_map_path,
    )
    from data import clean_text
    from feedback import feedback_training_rows
    from models import hstack_csr, write_hashes


# Configures logging for the training pipeline.
logging.basicConfig(
    filename=ROOT / "train.log",
    level=logging.INFO,
    format="%(asctime)s [INFO] %(message)s",
)


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

# Budget for the screening run that decides whether boosting or logistic
# regression earn a place beside the linear SVM.
#
# It is bounded deliberately.  Boosting builds one tree per category per round,
# and over a 180,000-column TF-IDF matrix the seven-category target alone ran
# past six minutes at 350 rounds — the twenty-seven category one would take
# hours, on a pipeline that has to be retrained whenever corrections
# accumulate.  The question the gate asks is therefore not "could this family
# win eventually" but "does it beat a calibrated linear SVM at a comparable
# cost", which is the question that decides what ships.  Raise these to give
# the challengers a longer run.
# Screening the challenger families is off by default.  Measured on a held-out
# split, the calibrated linear SVM matched the full three-model ensemble, while
# boosting over a 180,000-column TF-IDF matrix took longer than everything else
# in the pipeline combined — so paying for that comparison on every retrain buys
# a result already recorded in reports/benchmark.md.  Turn it on with
# `--gate`, or `"gate": {"enabled": true}`, when the comparison is wanted again.
GATE_ENABLED = bool(GATE.get("enabled", False))
GATE_BOOST_ROUNDS = int(GATE.get("max_boost_rounds", 120))
GATE_EARLY_STOPPING = int(GATE.get("early_stopping_rounds", 20))
GATE_MAX_ROWS = int(GATE.get("max_rows", 25_000))


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
    counts = pd.Series(y).value_counts()
    keep_classes = counts[counts >= 2].index
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

    # Calibration checks its own classes_ array with a heuristic meant for
    # spotting a regression target fed to a classifier: it warns when more
    # than half the values are unique and there are over 20 of them.  A
    # classes_ array is unique by definition, so any classifier with 21+
    # categories trips it once per fold.  It says nothing about the data.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%",
            category=UserWarning,
        )
        model.fit(X, y)

    return model


# Scores a fitted model's macro-F1 on the validation rows.
def macro_f1(model, X, y) -> float:
    """Macro-F1 of `model` on (X, y), or 0.0 when there is nothing to score.

    Macro rather than plain accuracy because the categories are wildly
    unbalanced — one Process category holds 23% of rows and several hold under
    0.1% — and accuracy is almost unmoved by getting the whole tail wrong.
    """
    if model is None or len(y) == 0:
        return 0.0
    return float(f1_score(y, model.predict(X), average="macro", zero_division=0))


# Decides which model families are worth keeping for a target.
def gate_models(target, X_train, y_train, X_val, y_val, num_labels) -> Tuple[dict, dict]:
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

    # The challengers are screened on a capped sample; the SVM that ships is
    # fitted on every labelled row.
    if not GATE_ENABLED:
        svm_only = fit_svm(X_train, y_train)
        if svm_only is None:
            raise RuntimeError(f"{target}: not enough labelled data to fit a classifier.")
        score = macro_f1(svm_only, X_val, y_val)
        info(f"  {target}: calibrated SVM macro-F1 {score:.4f} (challengers not screened)")
        return {"svm": svm_only}, {"svm": score}

    if len(y_train) > GATE_MAX_ROWS:
        rng = np.random.default_rng(SEED)
        sample = rng.choice(len(y_train), GATE_MAX_ROWS, replace=False)
        Xg, yg = X_train[sample], np.asarray(y_train)[sample]
        info(f"  {target}: screening challengers on {GATE_MAX_ROWS:,} of {len(y_train):,} rows")
    else:
        Xg, yg = X_train, np.asarray(y_train)

    svm_model = fit_svm(X_train, y_train)
    if svm_model is None:
        raise RuntimeError(f"{target}: not enough labelled data to fit a classifier.")

    baseline = macro_f1(svm_model, X_val, y_val)
    report["svm"] = baseline
    info(f"  {target}: calibrated SVM macro-F1 {baseline:.4f} (baseline)")

    kept = {"svm": svm_model}

    info(f"  {target}: Logistic Regression…")
    lr_model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.5, class_weight="balanced")
    lr_model.fit(Xg, yg)
    report["lr"] = macro_f1(lr_model, X_val, y_val)

    info(f"  {target}: XGBoost…")
    params = {
        "objective": "multi:softprob",
        "num_class": num_labels,
        "max_depth": 6,
        "learning_rate": 0.12,
        "eval_metric": "mlogloss",
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "reg_lambda": 1.0,
        "random_state": SEED,
    }
    booster = xgb.train(
        params,
        xgb.DMatrix(Xg, label=yg),
        num_boost_round=GATE_BOOST_ROUNDS,
        evals=[(xgb.DMatrix(X_val, label=y_val), "val")],
        early_stopping_rounds=GATE_EARLY_STOPPING,
        verbose_eval=False,
    )
    xgb_pred = booster.predict(xgb.DMatrix(X_val)).argmax(axis=1)
    report["xgb"] = float(f1_score(y_val, xgb_pred, average="macro", zero_division=0))

    for name, model in (("lr", lr_model), ("xgb", booster)):
        if report[name] > baseline:
            kept[name] = model
            info(f"  {target}: keeping {name} (macro-F1 {report[name]:.4f} > {baseline:.4f})")
        else:
            info(f"  {target}: dropping {name} (macro-F1 {report[name]:.4f} ≤ {baseline:.4f})")

    return kept, report


# Saves a target's models and records the weighting inference should use.
def save_models(out_dir: Path, models: dict, report: dict) -> None:
    """Write each kept model plus the ensemble.json describing the blend."""
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

    # Equal weights across whatever survived: the members are now all
    # calibrated probability outputs on the same scale, and there is no
    # held-out evidence for preferring one over another beyond the gate that
    # already admitted them.
    weights = {name: round(1.0 / len(models), 4) for name in models}
    (out_dir / "ensemble.json").write_text(
        json.dumps({"weights": weights, "validation_macro_f1": report}, indent=4)
    )
    write_hashes(written)


# Trains one classifier per parent category for a hierarchical target.
def train_hierarchical(
    target: str, spec: dict, label_maps: dict, hierarchy: dict,
    train_df, val_df, X_train, X_val,
) -> None:
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

        model = fit_svm(X_train[rows], tr_child[rows])
        if model is None:
            fallbacks[parent_name] = majority
            skipped += 1
            continue

        out_dir = out_root / str(parent_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / "svm.pkl")
        (out_dir / "ensemble.json").write_text(json.dumps({"weights": {"svm": 1.0}}, indent=4))
        write_hashes([out_dir / "svm.pkl"])
        fallbacks.setdefault(parent_name, majority)
        fitted += 1

        val_rows = (va_parent == parent_id) & (va_child >= 0)
        if val_rows.sum():
            scores.append(macro_f1(model, X_val[val_rows], va_child[val_rows]))

    (out_root / "routing.json").write_text(json.dumps({
        "parent": parent,
        "deterministic": deterministic,
        "fallback": fallbacks,
        "num_labels": num_labels,
    }, indent=4))

    mean_score = float(np.mean(scores)) if scores else 0.0
    info(
        f"  ✔ {target}: {fitted} per-parent model(s), {len(deterministic)} deterministic, "
        f"{skipped} too thin to fit; mean per-parent validation macro-F1 {mean_score:.4f}"
    )


# Runs the complete training pipeline.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate", action="store_true",
        help="Also fit logistic regression and boosting, and keep either only "
             "if it beats the calibrated SVM on validation macro-F1.",
    )
    args = parser.parse_args()

    global GATE_ENABLED
    GATE_ENABLED = GATE_ENABLED or args.gate

    info("Starting training…")

    if not TARGETS:
        raise ValueError(f"{CONFIG_PATH.name} is missing 'targets'.")

    clean_old_models()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROC_DIR / "train.csv")
    val_df = pd.read_csv(PROC_DIR / "val.csv")

    for df in (train_df, val_df):
        df["text"] = df["text"].astype(str).apply(clean_text)

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
            info(f"No reviewer corrections to fold into {target}.")
            continue

        extra = pd.DataFrame({
            "text": corrections["text"],
            **{f"label_{t}": (corrections["label"] if t == target else -1) for t in TARGETS},
        })
        train_df = pd.concat([train_df, extra], ignore_index=True)
        info(f"Added {len(extra)} reviewer correction(s) to {target}'s training rows.")

    # Trains TF-IDF vectorizers on training data.  These are unsupervised, so
    # one pair serves every target.
    info("Training TF-IDF vectorizers…")

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
    info(f"  feature matrix: {X_train.shape[0]:,} x {X_train.shape[1]:,}")

    joblib.dump(tfidf_word, MODEL_DIR / "tfidf_word.pkl")
    joblib.dump(tfidf_char, MODEL_DIR / "tfidf_char.pkl")
    write_hashes([MODEL_DIR / name for name in SHARED_ARTIFACTS])

    # Trains one set of classifiers per target, on the rows that target can
    # use.  A row missing this label carries -1 and is skipped here, while
    # still contributing to any other target it does have.
    for target, spec in TARGETS.items():
        col = f"label_{target}"
        tr_rows = train_df[col].to_numpy() >= 0
        va_rows = val_df[col].to_numpy() >= 0
        num_labels = len(label_maps[target])

        info(f"Training {target}: {int(tr_rows.sum()):,} rows, {num_labels} categories")

        if spec.get("parent"):
            train_hierarchical(
                target, spec, label_maps, hierarchy,
                train_df, val_df, X_train, X_val,
            )
            continue

        models, report = gate_models(
            target,
            X_train[tr_rows], train_df.loc[tr_rows, col].astype(int).to_numpy(),
            X_val[va_rows], val_df.loc[va_rows, col].astype(int).to_numpy(),
            num_labels,
        )
        save_models(MODEL_DIR / target, models, report)

    info("Training complete.")


if __name__ == "__main__":
    main()
