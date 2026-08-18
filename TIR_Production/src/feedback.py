#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.feedback
# Implements:
#   - REQ-014 (Reviewer corrections are recorded and reapplied)
#
# Corrections made in the web app land in data/feedback.csv and are used twice:
#
#   1. Immediately, through CorrectionIndex: a TIR whose wording matches one
#      that has already been corrected is answered with the reviewer's label
#      instead of the model's, so the same mistake is not repeated while the
#      current models are in place.
#   2. Permanently, by src.train, which folds the corrections into the training
#      data on the next run so the models themselves improve.
#
# Matching is exact on the cleaned text, and otherwise by cosine similarity of
# the same TF-IDF features the classifiers read.  The threshold is deliberately
# conservative: a wrong override is worse than no override, because it carries
# a reviewer's authority.

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

try:
    from src.data import clean_text
    from src.models import build_features
except ImportError:
    from data import clean_text
    from models import build_features


FEEDBACK_COLUMNS = ["timestamp", "target", "text", "predicted", "corrected"]

# Cosine similarity required before a stored correction is reused.
#
# Calibrated from both directions on the held-out split, scored with the same
# TF-IDF features the models read.
#
# Against wrong matches: different-category pairs sit far below it — their 99th
# percentile similarity is 0.086 — and at 0.80 only 1.9% of the pairs this would
# match were coded differently by a person.  Much of even that is not a matching
# failure, since coders gave conflicting categories to 31% of *identical*
# descriptions.
#
# Against missed matches, which is what fixes the threshold: measured on real
# edits to a stored wording, a typo scores 0.815, a plural 0.857 and an inserted
# word 0.842, while a reordering drops to 0.610 and a genuine rewording to 0.207.
# 0.80 therefore reuses a correction through the edits that leave a TIR the same
# TIR, and leaves anything genuinely reworded to the model.  At 0.90 none of
# those edits match at all and only an exact repeat is ever answered.
#
# The figures are not comparable to the 0.88 that the previous hash-embedding
# space used — TF-IDF similarity is on a different scale — but the behaviour is:
# that threshold also caught a typo and an inserted word while rejecting a
# rewording.
#
# Raise it towards 1.0 for stricter matching; 1.0 accepts exact matches only.
DEFAULT_SIMILARITY_THRESHOLD = 0.80


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
        tfidf_word: The fitted word-level vectorizer from training.
        tfidf_char: The fitted character-level vectorizer from training.
        threshold: Minimum cosine similarity for a non-exact match.

    Reusing the classifiers' own features means a correction is reapplied on
    the same notion of similarity the model was trained under, and removes the
    separate hash-embedding space that previously existed only for this.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        tfidf_word=None,
        tfidf_char=None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.tfidf_word = tfidf_word
        self.tfidf_char = tfidf_char
        self.threshold = float(threshold)
        self.texts: List[str] = [clean_text(t) for t in texts]
        self.labels: List[str] = [str(v) for v in labels]

        # Exact matches are answered from a dict; later entries win, matching
        # latest_corrections().
        self._exact = {t: v for t, v in zip(self.texts, self.labels)}

        self._matrix = None
        if self.texts and tfidf_word is not None and self.threshold <= 1.0:
            # Rows are unit-normalised once, so a lookup is a single sparse
            # dot product rather than a similarity computed per candidate.
            self._matrix = normalize(build_features(self.texts, tfidf_word, tfidf_char))

    # Finds the reviewer's answer for a text, if there is one.
    def lookup(self, text: str) -> Tuple[Optional[str], float]:
        """Return the corrected label for `text` and how closely it matched.

        Returns `(None, 0.0)` when nothing is close enough.
        """
        cleaned = clean_text(text)

        exact = self._exact.get(cleaned)
        if exact is not None:
            return exact, 1.0

        if self._matrix is None or self.threshold > 1.0:
            return None, 0.0

        vector = normalize(build_features([cleaned], self.tfidf_word, self.tfidf_char))
        # A text sharing no feature with anything stored embeds to all zeros,
        # which scores 0 against every row and cannot be anyone's match.
        similarities = (self._matrix @ vector.T).toarray().ravel()
        if not len(similarities):
            return None, 0.0

        best = int(np.argmax(similarities))
        score = float(similarities[best])
        return (self.labels[best], score) if score >= self.threshold else (None, 0.0)

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
