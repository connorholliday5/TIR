#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.train
# Trains TF-IDF models, deterministic dense embeddings, SVM, LR, and XGBoost for the ensemble pipeline.

import json
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.embedding import DEFAULT_MAX_CACHE, SubwordHashEmbedder
from src.feedback import feedback_training_rows
from src.paths import ROOT
from src.utils import clean_text, hash_file, hstack_csr


# Configures logging for the training pipeline.
logging.basicConfig(
    filename=ROOT / "train.log",
    level=logging.INFO,
    format="%(asctime)s [INFO] %(message)s",
)

# Prints a progress line and records it in the training log.
def info(msg):
    print(msg)
    logging.info(msg)


# Loads configuration values and prepares directories.
CONFIG = json.loads((ROOT / "config" / "config.json").read_text())
SEED = CONFIG.get("random_seed", 42)
TARGETS: dict = CONFIG.get("targets", {})

PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Vectorizers and embeddings are shared by every target; only the classifiers
# differ, so these live at the root of models/ and each target gets a subfolder.
SHARED_ARTIFACTS = ["tfidf_word.pkl", "tfidf_char.pkl", "dense_meta.json"]
TARGET_ARTIFACTS = ["svm.pkl", "lr.pkl", "xgb.json"]


# Configures the dense embedder shared by training and inference.
EMBEDDING_CFG = CONFIG.get("embedding", {})

# The same settings are written to models/dense_meta.json so that api.py,
# ensemble.py and the UI rebuild a byte-identical embedder at inference time.
EMBEDDER = SubwordHashEmbedder(
    dim=EMBEDDING_CFG.get("dim", 192),
    min_n=EMBEDDING_CFG.get("min_n", 3),
    max_n=EMBEDDING_CFG.get("max_n", 5),
    seed=EMBEDDING_CFG.get("seed", 0),
    l2_normalize=EMBEDDING_CFG.get("l2_normalize", False),
    max_cache=EMBEDDING_CFG.get("max_cache", DEFAULT_MAX_CACHE),
)


# Removes old model artifacts before training.
def clean_old_models():
    """Empty models/ so a run never mixes fresh artifacts with stale ones.

    Deepest paths first, so a folder is only removed once emptied.  Failures
    are reported rather than swallowed: a file that cannot be replaced would
    otherwise leave a previous run's model in place, and inference would load
    it against this run's label map.
    """
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


# Writes a SHA-256 digest beside each artifact.
def write_hashes(paths):
    for p in paths:
        (p.with_suffix(p.suffix + ".sha256")).write_text(hash_file(p))


# Trains the three classifiers for one target and saves them.
def train_target(target, label_map, X_train, y_train, X_val, y_val):
    """Fit the SVM, LR and XGBoost models for `target` on its labelled rows."""
    out_dir = MODEL_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)
    num_labels = len(label_map)

    # Calibration needs at least as many members per class as folds, so drop
    # any category too thin to fit and say which.  The label map keeps them,
    # so ids stay stable; those categories simply are never predicted.
    counts = y_train.value_counts()
    thin = sorted(counts[counts < 2].index)
    if thin:
        names = ", ".join(label_map[str(i)]["name"] for i in thin)
        info(f"  ⚠ {target}: {len(thin)} category(ies) with <2 training rows, not fitted: {names}")
        keep = ~y_train.isin(thin)
        X_train, y_train = X_train[keep.to_numpy()], y_train[keep]

    folds = int(min(3, y_train.value_counts().min()))
    if folds < 3:
        info(f"  ⚠ {target}: calibrating with cv={folds}; rarest category has {folds} training rows")

    info(f"  {target}: SVM...")
    svm_model = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000), method="sigmoid", cv=folds
    )

    # Calibration checks its own classes_ array with a heuristic meant for
    # spotting a regression target fed to a classifier: it warns when more
    # than half the values are unique and there are over 20 of them.  A
    # classes_ array is unique by definition, so any classifier with 21+
    # categories trips it once per fold.  It says nothing about the data, so
    # it is silenced here rather than left to alarm whoever reads the log.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%",
            category=UserWarning,
        )
        svm_model.fit(X_train, y_train)

    joblib.dump(svm_model, out_dir / "svm.pkl")

    info(f"  {target}: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=2500, solver="lbfgs", C=1.5)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, out_dir / "lr.pkl")

    info(f"  {target}: XGBoost...")
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
    bst = xgb.train(
        params,
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=350,
        evals=[(xgb.DMatrix(X_val, label=y_val), "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    bst.save_model(str(out_dir / "xgb.json"))

    write_hashes([out_dir / name for name in TARGET_ARTIFACTS])


# Runs the complete training pipeline.
def main():
    info("Starting training...")

    if not TARGETS:
        raise ValueError(
            "config/config.json is missing 'targets': it names the column to "
            "predict for each classification field."
        )

    clean_old_models()

    # Loads the preprocessed train/val/test datasets.
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    val_df   = pd.read_csv(PROC_DIR / "val.csv")
    test_df  = pd.read_csv(PROC_DIR / "test.csv")

    for df in (train_df, val_df, test_df):
        df["text"] = df["text"].astype(str).apply(clean_text)

    label_maps = {
        t: json.loads((DATA_DIR / f"label_map_{t}.json").read_text()) for t in TARGETS
    }

    # Folds reviewer corrections from the web app into the training split, so
    # the models themselves learn from the mistakes.  Corrections only ever
    # join training data: validation and test stay untouched so metrics remain
    # honest.  Each correction names the target it applies to.
    for target, label_map in label_maps.items():
        name2id = {v["name"]: int(k) for k, v in label_map.items()}
        corrections = feedback_training_rows(FEEDBACK_FILE, name2id, target=target)

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
    info("Training TF-IDF vectorizers...")

    tfidf_word = TfidfVectorizer(
        lowercase=True,
        strip_accents='ascii',
        ngram_range=(1,2),
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
        analyzer='char',
        ngram_range=(3,5),
        min_df=5,
        max_df=0.95,
        max_features=100_000,
        dtype=np.float32,  # type: ignore[arg-type]
        sublinear_tf=True,
    )

    X_train_tfidf = hstack_csr([
        tfidf_word.fit_transform(train_df["text"]),
        tfidf_char.fit_transform(train_df["text"]),
    ])
    X_val_tfidf = hstack_csr([
        tfidf_word.transform(val_df["text"]),
        tfidf_char.transform(val_df["text"]),
    ])

    joblib.dump(tfidf_word, MODEL_DIR / "tfidf_word.pkl")
    joblib.dump(tfidf_char, MODEL_DIR / "tfidf_char.pkl")

    # Generates deterministic dense embeddings, also shared by every target.
    info(f"Generating deterministic dense embeddings ({EMBEDDER!r})...")

    X_train = hstack_csr([X_train_tfidf, csr_matrix(EMBEDDER.embed_many(train_df["text"]))])
    X_val   = hstack_csr([X_val_tfidf,   csr_matrix(EMBEDDER.embed_many(val_df["text"]))])

    EMBEDDER.save_meta(MODEL_DIR / "dense_meta.json")
    EMBEDDER.clear_cache()

    write_hashes([MODEL_DIR / name for name in SHARED_ARTIFACTS])

    # Trains one set of classifiers per target, on the rows that target can
    # use.  A row missing this label carries -1 and is skipped here, while
    # still contributing to any other target it does have.
    for target, label_map in label_maps.items():
        col = f"label_{target}"
        tr = train_df[train_df[col] >= 0]
        va = val_df[val_df[col] >= 0]

        info(f"Training {target}: {len(tr):,} rows, {len(label_map)} categories")
        train_target(
            target,
            label_map,
            X_train[train_df[col].to_numpy() >= 0],
            tr[col].astype(int),
            X_val[val_df[col].to_numpy() >= 0],
            va[col].astype(int),
        )

    info("Training complete.")


if __name__ == "__main__":
    main()