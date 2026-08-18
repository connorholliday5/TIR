#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.feedback
# Implements:
#   - REQ-014 (Reviewer corrections are recorded and reapplied)
#
# Corrections made in the web app land in data/feedback.csv and are used twice:
#
#   1. Immediately, through CorrectionIndex: a TIR whose text matches one that
#      has already been corrected is answered with the reviewer's label instead
#      of the model's, so the same mistake is not repeated while the current
#      models are in place.
#   2. Permanently, by src.train, which folds the corrections into the training
#      data on the next run so the models themselves improve.
#
# Matching is exact on the cleaned text, and otherwise by cosine similarity of
# the dense embeddings, which catches near-duplicate wordings.  The threshold
# is deliberately conservative: a wrong override is worse than no override,
# because it carries a reviewer's authority.

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from src.embedding import SubwordHashEmbedder
from src.utils import clean_text


FEEDBACK_COLUMNS = ["timestamp", "target", "text", "predicted", "corrected"]

# Cosine similarity required before a stored correction is reused.
#
# Calibrated on 5.3M different-category pairs from the sample TIR export: the
# 99th percentile similarity is 0.23, and 0.88 leaves 14 pairs (0.0003%) able
# to match across categories.  On the other side, minor edits to a corrected
# text stay above it — one typo scores 0.94, an inserted word 0.92 — while a
# genuine rewording drops to 0.69 and is correctly left to the model.
#
# Raise it towards 1.0 for stricter matching; 1.0 accepts exact matches only.
DEFAULT_SIMILARITY_THRESHOLD = 0.88


# Appends one reviewer correction to the log.
def append_feedback(
    path: Path, text: str, predicted: str, corrected: str, target: str
) -> None:
    """Record a correction in the CSV at `path`, creating it if needed.

    `target` names the classification field the correction applies to, so a
    reviewer can fix one field without implying anything about the others.
    """
    row = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "text": clean_text(text),
        "predicted": predicted,
        "corrected": corrected,
    }])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a", index=False, header=not path.is_file())


# Reads the correction log.
def load_feedback(path: Path) -> pd.DataFrame:
    """Return the corrections stored at `path`.

    An absent or empty file yields an empty frame with the right columns, so
    callers never have to special-case a first run.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    # reindex both selects the expected columns and materialises any that are
    # absent, so a hand-edited file cannot raise a KeyError here.
    df = pd.read_csv(path).reindex(columns=FEEDBACK_COLUMNS)
    return df.dropna(subset=["text", "corrected"])


# Narrows the log to the corrections for one target.
def corrections_for(df: pd.DataFrame, target: str, default_target: str = "") -> pd.DataFrame:
    """Return the rows of `df` that correct `target`.

    Logs written before corrections carried a target have a blank one; they
    are attributed to `default_target`, which callers set to the primary
    target so older feedback is not silently discarded.
    """
    if df.empty:
        return df

    out = df.copy()
    out["target"] = out["target"].fillna("").astype(str).str.strip()
    if default_target:
        out.loc[out["target"] == "", "target"] = default_target
    return cast(pd.DataFrame, out.loc[out["target"] == target])


# Resolves each corrected text to the reviewer's final answer.
def latest_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the log to one row per text, keeping the most recent entry.

    A reviewer may correct the same wording twice; the later verdict wins.
    """
    if df.empty:
        return df

    out = df.copy()
    out["text"] = out["text"].astype(str).apply(clean_text)
    out = out.loc[out["text"].str.strip() != ""]
    return out.drop_duplicates(subset="text", keep="last")


# Answers a lookup with a reviewer's correction when one applies.
class CorrectionIndex:
    """Looks up reviewer corrections for a piece of text.

    Args:
        texts: Corrected TIR texts.
        labels: The reviewer's category for each text.
        embedder: Embedder used to compare near-duplicate wordings; the same
            one the models were trained with.
        threshold: Minimum cosine similarity for a non-exact match.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        embedder: SubwordHashEmbedder,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.embedder = embedder
        self.threshold = float(threshold)
        self.texts: List[str] = [clean_text(t) for t in texts]
        self.labels: List[str] = [str(v) for v in labels]

        # Exact matches are answered from a dict; later entries win, matching
        # latest_corrections().
        self._exact = {t: v for t, v in zip(self.texts, self.labels)}

        if self.texts and self.threshold <= 1.0:
            vectors = embedder.embed_many(self.texts)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # A zero-length text embeds to the zero vector; leave it at zero so
            # it can never be the nearest neighbour of anything.
            self._unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        else:
            self._unit = np.zeros((0, embedder.dim), dtype=np.float32)

    # Builds an index from the correction log on disk.
    @classmethod
    def from_file(
        cls,
        path: Path,
        embedder: SubwordHashEmbedder,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> "CorrectionIndex":
        """Load `path` and build an index from its most recent corrections."""
        df = latest_corrections(load_feedback(path))
        return cls(df["text"].tolist(), df["corrected"].tolist(), embedder, threshold)

    # Finds the reviewer's answer for a text, if there is one.
    def lookup(self, text: str) -> Tuple[Optional[str], float]:
        """Return the corrected label for `text` and how closely it matched.

        Returns `(None, 0.0)` when nothing is close enough.
        """
        cleaned = clean_text(text)

        exact = self._exact.get(cleaned)
        if exact is not None:
            return exact, 1.0

        if not len(self._unit) or self.threshold > 1.0:
            return None, 0.0

        vec = self.embedder.embed(cleaned)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return None, 0.0

        sims = self._unit @ (vec / norm)
        best = int(np.argmax(sims))
        score = float(sims[best])

        if score >= self.threshold:
            return self.labels[best], score
        return None, 0.0

    def __len__(self) -> int:
        return len(self._exact)


# Turns corrections into extra labelled training rows.
def feedback_training_rows(
    path: Path, name2id: dict, target: str = "", default_target: str = ""
) -> pd.DataFrame:
    """Return corrections as `text`/`label` rows ready to append to training.

    Corrections naming a category that is not in the label map are dropped:
    the map is rebuilt by src.preprocess, and a new category cannot be learned
    without regenerating it.

    Args:
        path: The feedback CSV.
        name2id: Category name -> label id for the target being trained.
        target: Only use corrections for this field; empty means all of them.
        default_target: Field to attribute corrections logged without one.

    Returns:
        A DataFrame with `text` and `label` columns (possibly empty).
    """
    df = load_feedback(Path(path))
    if target:
        df = corrections_for(df, target, default_target or target)
    df = latest_corrections(df)
    if df.empty:
        return pd.DataFrame(columns=["text", "label"])

    known = df[df["corrected"].isin(name2id)]
    return pd.DataFrame({
        "text": known["text"].tolist(),
        "label": [int(name2id[v]) for v in known["corrected"]],
    })