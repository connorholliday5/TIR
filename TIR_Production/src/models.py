#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.models
# Feature construction, the arithmetic that turns model outputs into one
# answer per row, and artifact integrity checks.

import hashlib
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, cast
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


# Stacks sparse blocks side by side and returns a CSR matrix.
def hstack_csr(blocks: Sequence[Any]) -> csr_matrix:
    """Horizontally stack `blocks` into a single CSR matrix.

    `format="csr"` already guarantees the result, but scipy annotates
    hstack/vstack as returning a union of every sparse container it knows
    about, one of which declares `.shape` as optional.  Narrowing here keeps
    that union from leaking into sklearn's `fit`, which only accepts concrete
    matrix types.
    """
    return cast(csr_matrix, hstack(blocks, format="csr"))


# Stacks sparse blocks on top of each other and returns a CSR matrix.
def vstack_csr(blocks: Sequence[Any]) -> csr_matrix:
    """Vertically stack `blocks` into a single CSR matrix."""
    return cast(csr_matrix, vstack(blocks, format="csr"))


# Builds the feature matrix for a batch of texts.
def build_features(texts: Sequence[str], tfidf_word, tfidf_char) -> csr_matrix:
    """Encode `texts` as the word- and character-level TF-IDF the models read.

    A dense subword-hash block used to be appended here.  It assigned every
    subword a random vector derived from a SHA-256 digest, so unlike the
    trained FastText vectors it replaced it carried no relationship between
    words, and its character 3-5 grams covered the same span `tfidf_char`
    already encodes exactly.  Measured on a held-out split it cost 0.1 points
    of accuracy and 0.3 of macro-F1 while taking roughly sixteen times as long
    to compute as training the model itself.

    Transforming a whole column in one call rather than a row at a time is
    what keeps a 94,000-row export practical.
    """
    listed = [str(t) for t in texts]
    return hstack_csr([tfidf_word.transform(listed), tfidf_char.transform(listed)])


# Scatters a model's probabilities into the full label space.
def expand_proba(proba: np.ndarray, classes: np.ndarray, num_labels: int) -> np.ndarray:
    """Widen `proba` so its columns line up with label ids 0..num_labels-1.

    scikit-learn returns one column per class it actually saw while fitting.
    When a category is defined in the label map but absent from the training
    split — which happens to rare categories, and to every category outside a
    per-parent model's own children — those columns are missing, and blending
    the outputs would silently align the wrong categories.

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


# Blends the per-model probabilities into one score per category.
def ensemble_score_matrix(
    probas: Dict[str, np.ndarray], weights: Dict[str, float], num_labels: int
) -> np.ndarray:
    """Weight and sum each model's probabilities into one (n_rows, n_labels).

    Every model contributes a probability distribution, so the weighted sum is
    itself a probability once the weights are normalised.  That is what lets
    `review_threshold` mean "at least this likely", and what lets the coverage
    curve in the benchmark be read as a precision guarantee.

    The previous form one-hot encoded the SVM's hard prediction and gave it a
    flat 0.40, which put a floor of 0.40 under the winning score and a ceiling
    of 0.60 over every rival — so the number reported as "confidence" could not
    be compared against a probability at all.  The SVM is calibrated at
    training time precisely so it can supply `predict_proba` here.
    """
    used = {
        name: matrix for name, matrix in probas.items()
        if float(weights.get(name, 0.0)) > 0.0
    }
    if not used:
        raise ValueError(
            "No model has a non-zero ensemble weight; nothing can be predicted. "
            f"Weights were: {weights}"
        )

    total = sum(float(weights[name]) for name in used)
    n_rows = len(next(iter(used.values())))

    combined = np.zeros((n_rows, num_labels), dtype=np.float32)
    for name, matrix in used.items():
        block = np.asarray(matrix, dtype=np.float32).reshape(n_rows, num_labels)
        combined += (float(weights[name]) / total) * block

    return combined


# Reduces the blended scores to one prediction per row.
def combine_ensemble_scores(
    probas: Dict[str, np.ndarray], weights: Dict[str, float], num_labels: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (label_ids, confidences), the confidence being a probability."""
    combined = ensemble_score_matrix(probas, weights, num_labels)
    return combined.argmax(axis=1), combined.max(axis=1)


# Ranks the best few categories for each row.
def top_k_from_scores(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the `k` highest-scoring label ids per row and their scores.

    Used where a single answer would overstate what the data supports: the
    deepest Process level is coded consistently by people only about two-thirds
    of the time, so the model offers a short ranked list to choose from rather
    than one label presented as the answer.

    Returns:
        (ids, scores), each (n_rows, min(k, n_labels)), ordered best first.
    """
    k = max(1, min(int(k), scores.shape[1]))
    # argpartition finds the top k without sorting the whole row, then only
    # those k are sorted; over a 543-category label space that is the
    # difference between sorting everything and sorting three items.
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1)
    ordered = np.take_along_axis(idx, order, axis=1)
    return ordered, np.take_along_axis(scores, ordered, axis=1)


# Returns the SHA-256 digest of a file.
def hash_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


# Writes a SHA-256 digest beside each artifact.
def write_hashes(paths: Sequence[Path]) -> None:
    """Record the digest of each artifact for later verification."""
    for path in paths:
        path.with_suffix(path.suffix + ".sha256").write_text(hash_file(path))


# Checks a model artifact against its stored digest.
def verify_model_hash(model_path: Path) -> None:
    """Check that the model file matches its stored *.sha256* digest.

    Verification runs before joblib.load, which unpickles and therefore
    executes; it is only meaningful while the files are still untouched bytes
    on disk.
    """
    model_path = Path(model_path)
    hash_path = model_path.with_suffix(model_path.suffix + ".sha256")
    if not hash_path.is_file():
        # No hash file - skip verification (optional safety net)
        return

    expected = hash_path.read_text().strip()
    actual = hash_file(model_path)
    if actual != expected:
        raise FileNotFoundError(
            f"Model integrity check failed for {model_path.name}: "
            f"expected {expected[:8]}…, got {actual[:8]}…"
        )
