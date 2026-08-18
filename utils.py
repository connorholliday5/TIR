#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.utils
# Author: chollida
# Created: 2026-06-29
# Last Modified: 2026-07-22
# Implements:
#   - REQ-002 (Text cleaning)
#   - REQ-010 (Feature matrix construction)
#   - REQ-011 (Input validation)
#   - REQ-012 (Model integrity verification)

import hashlib
import re
from pathlib import Path
from typing import Any, List, Sequence, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack

from src.embedding import SubwordHashEmbedder


# Stacks sparse blocks side by side and returns a CSR matrix.
def hstack_csr(blocks: Sequence[Any]) -> csr_matrix:
    """Horizontally stack `blocks` into a single CSR matrix.

    `format="csr"` already guarantees the result, but scipy annotates
    hstack/vstack as returning a union of every sparse container it knows
    about, one of which declares `.shape` as optional.  Narrowing here keeps
    that union from leaking into every caller (and into sklearn's `fit`,
    which only accepts concrete matrix types).
    """
    return cast(csr_matrix, hstack(blocks, format="csr"))


# Stacks sparse blocks on top of each other and returns a CSR matrix.
def vstack_csr(blocks: Sequence[Any]) -> csr_matrix:
    """Vertically stack `blocks` into a single CSR matrix."""
    return cast(csr_matrix, vstack(blocks, format="csr"))


# Scatters a model's probabilities into the full label space.
def expand_proba(proba: np.ndarray, classes: np.ndarray, num_labels: int) -> np.ndarray:
    """Widen `proba` so its columns line up with label ids 0..num_labels-1.

    scikit-learn returns one column per class it actually saw while fitting.
    When a category is defined in the label map but absent from the training
    split — which happens to rare categories — those columns are missing, and
    blending the outputs would silently align the wrong categories.

    Args:
        proba: Probabilities of shape (n_rows, len(classes)).
        classes: The model's `classes_`, i.e. the label id of each column.
        num_labels: Size of the full label space.

    Returns:
        An (n_rows, num_labels) array, zero for categories never trained on.
    """
    proba = np.asarray(proba, dtype=np.float32)
    classes = np.asarray(classes, dtype=int)

    if proba.shape[1] == num_labels and np.array_equal(classes, np.arange(num_labels)):
        return proba

    out = np.zeros((proba.shape[0], num_labels), dtype=np.float32)
    out[:, classes] = proba
    return out


# Reports whether a text has anything left to classify.
def is_blank_text(text) -> bool:
    """True when `text` is empty, missing, or only whitespace.

    A blank description still produces a prediction, because the models score
    an all-zero feature vector rather than refusing, and that prediction can
    look confident: an empty string scored 99.8% for one target.  Callers use
    this to withhold the answer instead of presenting a category nothing was
    read from.
    """
    return not str(text).strip() or str(text).strip().lower() in {"nan", "none"}


# Blends the three model outputs into a single prediction per row.
def combine_ensemble_scores(
    svm_pred: np.ndarray,
    lr_proba: np.ndarray,
    xgb_proba: np.ndarray,
    weights: dict,
    num_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weight and combine the SVM, LR and XGBoost outputs.

    The SVM is a hard classifier, so its prediction is one-hot encoded before
    being weighted against the two probability outputs.  Lives here so the
    API, the batch pipeline and the UI cannot drift apart on the arithmetic.

    Args:
        svm_pred: Predicted label per row, shape (n_rows,).
        lr_proba: Logistic Regression probabilities, shape (n_rows, n_labels).
        xgb_proba: XGBoost probabilities, shape (n_rows, n_labels).
        weights: Per-model weights, keyed "xgb", "svm" and "lr".
        num_labels: Total number of categories.

    Returns:
        A (label_ids, confidences) pair, each of shape (n_rows,).
    """
    n_rows = len(svm_pred)

    svm_onehot = np.zeros((n_rows, num_labels), dtype=np.float32)
    svm_onehot[np.arange(n_rows), np.asarray(svm_pred, dtype=int)] = 1.0

    combined = (
        weights["xgb"] * np.asarray(xgb_proba, dtype=np.float32).reshape(n_rows, num_labels) +
        weights["svm"] * svm_onehot +
        weights["lr"] * np.asarray(lr_proba, dtype=np.float32).reshape(n_rows, num_labels)
    )

    return combined.argmax(axis=1), combined.max(axis=1)


# Standardises raw description text.
def clean_text(text: str) -> str:
    """Standardise raw description text.

    Collapses every run of whitespace — spaces, tabs, new-lines — to a single
    space, then trims and lower-cases.

    The docstring previously promised "extra spaces" were removed but the code
    only substituted tabs and new-lines for spaces, so a doubled space
    survived.  Character n-grams see that gap, and 2% of the sample export
    contains one, meaning those records classified differently for a reason
    that carries no meaning.
    """
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# Joins the configured description columns into the model's input text.
def build_text_series(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Join `columns` into the single text field the models are trained on.

    Driven by `required_columns` in config.json, so training, batch inference
    and the UI always compose their input the same way.

    `fillna` runs before `astype`: on pandas 2.x `astype(str)` renders a
    missing value as the literal string "nan", which `fillna` can no longer
    replace, appending a phantom token to every row.

    Args:
        df: Frame containing the description columns.
        columns: Column names to join, in order.

    Returns:
        A Series of cleaned text, one entry per row.
    """
    if not columns:
        raise ValueError("At least one description column is required.")

    # Subscripting a DataFrame is typed as returning either a Series or a
    # DataFrame; the column names here always select a single Series.
    def column(name: str) -> pd.Series:
        return cast(pd.Series, df[name]).fillna("").astype(str)

    combined = column(columns[0])
    for name in columns[1:]:
        combined = combined + " " + column(name)

    # Series.apply is typed as possibly returning a DataFrame, which it only
    # does when the applied function itself returns a Series.
    return cast(pd.Series, combined.apply(clean_text))


# Maps raw category values onto their standard form.
def normalize_categories(
    values: pd.Series, table: dict, unknown: str = ""
) -> pd.Series:
    """Map raw category values onto their standard form.

    Used by preprocessing to build labels and by inference to compare
    predictions against ground truth, so both sides read a raw export the same
    way.

    Args:
        values: Raw column values.
        table: Lower-case value -> standard form. Empty leaves values as-is.
        unknown: Standard form for values absent from `table`; when empty they
            become NA so the caller can treat them as unlabelled.

    Returns:
        A Series of standardised names.
    """
    out = cast(pd.Series, values).astype(str).str.strip()
    if not table:
        return out

    out = out.str.lower().map(table)
    return out.fillna(unknown) if unknown else out


# Checks that every column the pipeline needs is present.
def validate_input_dataframe(df: pd.DataFrame, required: List[str]) -> None:
    """Raise a ValueError if any of the required columns are missing.

    Args:
        df: DataFrame supplied by the caller.
        required: List of column names that must exist.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required column(s): {', '.join(missing)}"
        )


# Builds the combined sparse feature vector for one text.
def build_feature_matrix(
    text: str,
    tfidf_word,
    tfidf_char,
    embedder: SubwordHashEmbedder,
) -> csr_matrix:
    """Create the combined sparse feature vector for a single text string.

    The function mirrors the logic used in `api`, `ensemble` and the UI
    but lives in a single place so that any change is automatically
    propagated everywhere.

    Args:
        text: Cleaned text to encode.
        tfidf_word: Fitted word-level TF-IDF vectorizer.
        tfidf_char: Fitted character-level TF-IDF vectorizer.
        embedder: Dense embedder rebuilt from `models/dense_meta.json`.

    Returns:
        A CSR matrix of shape (1, n_features).
    """
    # TF‑IDF part
    Xw = tfidf_word.transform([text])
    Xc = tfidf_char.transform([text])

    # Dense subword part (dense → CSR)
    dense = csr_matrix(embedder.embed(text).reshape(1, -1))

    # Concatenate both representations, in the same column order as training
    return hstack_csr([Xw, Xc, dense])


# Returns the SHA-256 digest of a file.
def hash_file(path: Path) -> str:
    """Return the hex SHA‑256 digest of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


# Checks a model artifact against its stored digest.
def verify_model_hash(model_path: Path) -> None:
    """Check that the model file matches the stored *.sha256* digest.

    The companion file must be named `<model>.sha256` and contain a
    single line with the expected hex digest.
    """
    hash_path = model_path.with_suffix(model_path.suffix + ".sha256")
    if not hash_path.is_file():
        # No hash file – skip verification (optional safety net)
        return

    expected = hash_path.read_text().strip()
    actual = hash_file(model_path)
    if actual != expected:
        raise FileNotFoundError(
            f"Model integrity check failed for {model_path.name}: "
            f"expected {expected[:8]}…, got {actual[:8]}…"
        )