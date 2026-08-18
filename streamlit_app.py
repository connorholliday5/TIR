#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.streamlit_app
# Web UI for classifying Technical Issue Reports.
#
# Run from the repository root:
#     streamlit run src/streamlit_app.py

import sys
from pathlib import Path

# `streamlit run` executes this file as a plain script rather than as part of
# the `src` package, so the repository root is not on sys.path and the imports
# below would fail with "ModuleNotFoundError: No module named 'src'".  This has
# to run before any src import.
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import json
from typing import cast

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

from src.embedding import SubwordHashEmbedder
from src.feedback import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CorrectionIndex,
    append_feedback,
    corrections_for,
    latest_corrections,
    load_feedback,
)
from src.paths import ROOT
from src.utils import (
    build_feature_matrix,
    build_text_series,
    clean_text,
    combine_ensemble_scores,
    expand_proba,
    is_blank_text,
    validate_input_dataframe,
    verify_model_hash,
    vstack_csr,
)


# Loads configuration shared with the training and batch pipelines.
CONFIG = json.loads((ROOT / "config" / "config.json").read_text())
WEIGHTS = CONFIG.get("ensemble_weights", {"xgb": 0.40, "svm": 0.40, "lr": 0.20})
REQUIRED_COLS = CONFIG.get("required_columns", ["Description 1"])
REVIEW_THRESHOLD = CONFIG.get("review_threshold", 0.55)
TARGETS: dict = CONFIG.get("targets", {})
PRIMARY_TARGET = CONFIG.get("primary_target", next(iter(TARGETS), ""))

FEEDBACK_CFG = CONFIG.get("feedback", {})
FEEDBACK_ENABLED = FEEDBACK_CFG.get("enabled", False)
SIMILARITY_THRESHOLD = FEEDBACK_CFG.get(
    "similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD
)

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

SHARED_ARTIFACTS = ["tfidf_word.pkl", "tfidf_char.pkl", "dense_meta.json"]
TARGET_ARTIFACTS = ["svm.pkl", "lr.pkl", "xgb.json"]


# Turns a target key into something readable, e.g. "process_cat" -> "Process Cat".
def target_title(target: str) -> str:
    """Human-facing name for a target, taken from its source column."""
    return TARGETS.get(target, {}).get("column", target.replace("_", " ").title())


# Loads every model artifact once per session.
@st.cache_resource(show_spinner="Loading models…")
def load_bundle() -> dict:
    """Verify and load the shared encoders and each target's classifiers.

    Raises:
        FileNotFoundError: If an artifact is missing, or if one fails its
            SHA-256 integrity check.
    """
    missing = [n for n in SHARED_ARTIFACTS if not (MODEL_DIR / n).is_file()]
    for target in TARGETS:
        missing += [
            f"{target}/{n}" for n in TARGET_ARTIFACTS
            if not (MODEL_DIR / target / n).is_file()
        ]
        if not (DATA_DIR / f"label_map_{target}.json").is_file():
            missing.append(f"data/label_map_{target}.json")
    if missing:
        raise FileNotFoundError(", ".join(missing))

    # Integrity first: joblib.load unpickles, which executes.
    for name in SHARED_ARTIFACTS:
        verify_model_hash(MODEL_DIR / name)

    bundle = {
        "tfidf_word": joblib.load(MODEL_DIR / "tfidf_word.pkl"),
        "tfidf_char": joblib.load(MODEL_DIR / "tfidf_char.pkl"),
        "embedder": SubwordHashEmbedder.load(MODEL_DIR / "dense_meta.json"),
        "targets": {},
    }

    for target in TARGETS:
        target_dir = MODEL_DIR / target
        for name in TARGET_ARTIFACTS:
            verify_model_hash(target_dir / name)

        booster = xgb.Booster()
        booster.load_model(str(target_dir / "xgb.json"))

        raw = json.loads((DATA_DIR / f"label_map_{target}.json").read_text())
        bundle["targets"][target] = {
            "svm": joblib.load(target_dir / "svm.pkl"),
            "lr": joblib.load(target_dir / "lr.pkl"),
            "xgb": booster,
            "id2name": {int(k): v["name"] for k, v in raw.items()},
        }

    return bundle


# Rebuilds a target's correction index whenever the feedback file changes.
@st.cache_resource(show_spinner=False)
def load_corrections(
    _embedder: SubwordHashEmbedder, target: str, signature: tuple
) -> CorrectionIndex:
    """Build a CorrectionIndex for `target`, keyed on the file's signature.

    `signature` is the file's (mtime, size); passing it as an argument is what
    makes Streamlit rebuild the index as soon as a correction is saved.  The
    embedder is prefixed with an underscore so Streamlit does not try to hash
    it.
    """
    if not FEEDBACK_ENABLED:
        return CorrectionIndex([], [], _embedder, SIMILARITY_THRESHOLD)

    log = corrections_for(load_feedback(FEEDBACK_PATH), target, PRIMARY_TARGET)
    active = latest_corrections(log)
    return CorrectionIndex(
        active["text"].tolist(),
        active["corrected"].tolist(),
        _embedder,
        SIMILARITY_THRESHOLD,
    )


# Returns the current feedback-file signature, used as a cache key.
def feedback_signature() -> tuple:
    """Return (mtime, size) for the feedback file, or zeros when absent."""
    if not FEEDBACK_PATH.is_file():
        return (0.0, 0)
    stat = FEEDBACK_PATH.stat()
    return (stat.st_mtime, stat.st_size)


# Classifies a list of cleaned texts against every configured target.
def classify(texts: list, bundle: dict) -> pd.DataFrame:
    """Return one row per text with a prediction column set per target.

    Uses exactly the same feature construction and ensemble weighting as
    `src.ensemble` and `src.api`, then lets any reviewer correction override
    the result when that is switched on.
    """
    X = vstack_csr([
        build_feature_matrix(t, bundle["tfidf_word"], bundle["tfidf_char"], bundle["embedder"])
        for t in texts
    ])

    out = pd.DataFrame(index=range(len(texts)))

    for target, models in bundle["targets"].items():
        id2name = models["id2name"]
        num_labels = len(id2name)

        ids, confidence = combine_ensemble_scores(
            models["svm"].predict(X),
            expand_proba(models["lr"].predict_proba(X), models["lr"].classes_, num_labels),
            models["xgb"].predict(xgb.DMatrix(X)),
            WEIGHTS,
            num_labels,
        )

        out[f"pred_{target}"] = [id2name.get(int(i), "") for i in ids]
        out[f"confidence_{target}"] = confidence
        out[f"review_{target}"] = confidence < REVIEW_THRESHOLD
        out[f"source_{target}"] = "model"

        # A row with no description gets no answer.  The models still score an
        # all-zero vector and can report high confidence doing it.
        for row, text in enumerate(texts):
            if is_blank_text(text):
                out.at[row, f"pred_{target}"] = ""
                out.at[row, f"confidence_{target}"] = 0.0
                out.at[row, f"review_{target}"] = True
                out.at[row, f"source_{target}"] = "no description"

        # A reviewer has already ruled on this wording: use their answer.
        index = load_corrections(bundle["embedder"], target, feedback_signature())
        if not len(index):
            continue

        for row, text in enumerate(texts):
            label, score = index.lookup(text)
            if label is None or label == out.at[row, f"pred_{target}"]:
                continue
            out.at[row, f"pred_{target}"] = label
            out.at[row, f"confidence_{target}"] = score
            out.at[row, f"review_{target}"] = False
            out.at[row, f"source_{target}"] = "reviewer correction"

    return out


# Renders one target's prediction plus its correction control.
def render_prediction(target: str, result: dict, bundle: dict) -> None:
    label = result[f"pred_{target}"]
    confidence = result[f"confidence_{target}"]
    from_reviewer = result.get(f"source_{target}") == "reviewer correction"

    st.markdown(f"#### {target_title(target)}")

    left, right = st.columns(2)
    left.metric("Predicted category", label or "—")
    right.metric("Match" if from_reviewer else "Confidence", f"{confidence:.1%}")

    if from_reviewer:
        st.info(
            "Answered from a previous reviewer correction, not the model. "
            "Run `python -m src.train` to fold it into the models themselves."
        )
    elif result[f"review_{target}"]:
        st.warning(f"Confidence is below {REVIEW_THRESHOLD:.0%} — worth a human review.")

    categories = sorted(bundle["targets"][target]["id2name"].values())
    choice = st.selectbox(
        "Correct category",
        categories,
        index=categories.index(label) if label in categories else 0,
        key=f"correction_{target}",
    )
    if st.button("Save correction", key=f"save_{target}"):
        # st.selectbox is typed as possibly returning None (it does when the
        # option list is empty), which cannot happen here.
        if choice is None or choice == label:
            st.info("That matches the prediction — nothing to log.")
        else:
            append_feedback(
                FEEDBACK_PATH, st.session_state["result_text"], label, choice, target
            )
            st.session_state["result"] = classify(
                [st.session_state["result_text"]], bundle
            ).iloc[0].to_dict()
            saved_to = FEEDBACK_PATH.relative_to(ROOT)
            # Survives the rerun below, which would otherwise discard it.
            st.session_state["flash"] = (
                f"Saved to {saved_to} — this wording now returns **{choice}** "
                f"for {target_title(target)}."
                if FEEDBACK_ENABLED else
                f"Saved to {saved_to} — it will be folded into the models on the "
                f"next `python -m src.train`."
            )
            st.rerun()


# Renders the single-TIR tab.
def render_single(bundle: dict) -> None:
    st.subheader("Classify a single TIR")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    description = st.text_area("Description", height=160, key="description")

    if st.button("Classify", type="primary"):
        if not description.strip():
            st.warning("Enter a description first.")
            return
        text = clean_text(description)
        st.session_state["result"] = classify([text], bundle).iloc[0].to_dict()
        st.session_state["result_text"] = text

    result = st.session_state.get("result")
    if not result:
        return

    st.caption(
        "Wrong category? Pick the right one — it takes effect immediately for "
        "this wording and is folded into the models on the next retrain."
        if FEEDBACK_ENABLED else
        "Wrong category? Pick the right one — it is logged and folded into the "
        "models on the next retrain."
    )

    targets = bundle["targets"]
    columns = st.columns(len(targets)) if len(targets) > 1 else [st.container()]
    for column, target in zip(columns, targets):
        with column:
            render_prediction(target, result, bundle)


# Chooses the columns worth putting in front of a reviewer.
def display_columns(out: pd.DataFrame, targets: list) -> list:
    """Return the readable subset of a predictions frame, in reading order.

    The description first, then each target's category, confidence and review
    flag.  Columns the uploaded file happened to carry are left out, as are
    the pipeline's own working columns: `text` duplicates the description, and
    `label_*` are the numeric ids preprocessing assigns.  `source_*` is only
    interesting once a reviewer correction has actually overridden something.
    """
    columns = [c for c in REQUIRED_COLS if c in out.columns]

    for target in targets:
        columns += [f"pred_{target}", f"confidence_{target}", f"review_{target}"]
        source = f"source_{target}"
        if source in out.columns and (out[source] != "model").any():
            columns.append(source)

    return [c for c in columns if c in out.columns]


# Labels and formats those columns for display.
def display_format(targets: list) -> dict:
    """Column titles and formats for the reviewer's table."""
    config = {
        column: st.column_config.TextColumn("Description", width="large")
        for column in REQUIRED_COLS
    }

    for target in targets:
        title = target_title(target)
        config[f"pred_{target}"] = st.column_config.TextColumn(title, width="medium")
        # "percent" scales the 0-1 value for display; a printf format such as
        # "%.0f%%" would print 0.47 as "0%".
        config[f"confidence_{target}"] = st.column_config.ProgressColumn(
            "Confidence", format="percent", min_value=0.0, max_value=1.0
        )
        config[f"review_{target}"] = st.column_config.CheckboxColumn("Review?")
        config[f"source_{target}"] = st.column_config.TextColumn("Source")

    return config


# Renders the batch-upload tab.
def render_batch(bundle: dict) -> None:
    st.subheader("Classify a file")
    st.caption(
        f"CSV or Excel containing {' and '.join(REQUIRED_COLS)}. "
        "Any other columns are carried through untouched."
    )

    upload = st.file_uploader("TIR export", type=["csv", "xlsx", "xls"])
    if upload is None:
        return

    df = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)

    try:
        validate_input_dataframe(df, REQUIRED_COLS)
    except ValueError as exc:
        st.error(str(exc))
        return

    texts = build_text_series(df, REQUIRED_COLS)

    with st.spinner(f"Classifying {len(df)} rows…"):
        preds = classify(texts.tolist(), bundle)

    out = pd.concat([df.reset_index(drop=True), preds], axis=1)
    targets = list(bundle["targets"])

    # Headline counts first: how much of the file a coder still has to touch.
    # any(axis=1) keeps this a boolean Series carrying the frame's own index, so
    # the .loc filters below stay aligned; going through numpy would hand .loc a
    # bare array and lose that alignment.  The cast is only for the type checker,
    # whose pandas stubs disagree about what axis=1 returns.
    needs_review = cast(
        pd.Series, preds[[f"review_{t}" for t in targets]].any(axis=1)
    )
    cells = st.columns(1 + len(targets))
    cells[0].metric("Rows classified", f"{len(out):,}")
    for cell, target in zip(cells[1:], targets):
        flagged = int(preds[f"review_{target}"].to_numpy().sum())
        cell.metric(
            f"{target_title(target)} to review",
            f"{flagged:,}",
            f"{flagged / len(out):.0%} of rows" if len(out) else None,
            delta_color="off",
        )

    left, right = st.columns(2)
    only_review = left.toggle(
        "Only rows needing review",
        help="Hide the rows the models were confident about.",
    )
    show_everything = right.toggle(
        "Show every column",
        help="Include the uploaded file's other columns and the internal ones.",
    )

    view = out.loc[needs_review] if only_review else out
    if view.empty:
        st.info("Nothing was flagged for review — every row cleared the threshold.")
        return

    st.dataframe(
        view if show_everything else view[display_columns(view, targets)],
        column_config=None if show_everything else display_format(targets),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download predictions.csv",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        help="The full file: everything uploaded, plus every prediction column.",
    )
    if needs_review.any():
        st.download_button(
            f"Download review queue ({int(needs_review.sum()):,} rows)",
            out.loc[needs_review].to_csv(index=False).encode("utf-8"),
            file_name="review_queue.csv",
            mime="text/csv",
            help="Only the rows a coder still needs to look at.",
        )


# Renders the saved-corrections tab.
def render_corrections(bundle: dict) -> None:
    st.subheader("Reviewer corrections")

    log = load_feedback(FEEDBACK_PATH)
    if log.empty:
        st.info(
            "No corrections yet. Correct a prediction on the **Single TIR** tab "
            "and it will appear here."
        )
        return

    active = latest_corrections(log)
    st.caption(
        f"{len(log)} correction(s) logged, {len(active)} distinct wording(s) in "
        f"effect. Matching is exact, or above {SIMILARITY_THRESHOLD:.0%} "
        "similarity for near-duplicate wordings."
        if FEEDBACK_ENABLED else
        f"{len(log)} correction(s) logged, {len(active)} distinct wording(s). "
        "These are not applied at prediction time — set feedback.enabled in "
        "config.json to turn that on — but every one is used the next time the "
        "models are trained."
    )

    st.dataframe(log.iloc[::-1], use_container_width=True)
    st.download_button(
        "Download feedback.csv",
        log.to_csv(index=False).encode("utf-8"),
        file_name="feedback.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown(
        ("These corrections are applied on top of the model right now. To teach "
         "the models themselves, retrain — " if FEEDBACK_ENABLED else
         "To teach the models these corrections, retrain — ")
        + "`src.train` folds every correction into the training split:\n"
        "```\n"
        "python -m src.train\n"
        "```"
    )


# Builds the page.
def main() -> None:
    st.set_page_config(page_title="TIR Classification", page_icon="🔧", layout="wide")
    st.title("TIR Classification System")

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        st.error(f"Models are not ready: {exc}")
        st.markdown(
            "Run the pipeline first, from the repository root:\n"
            "```\n"
            "python -m src.preprocess --raw_csv \"<your export>.xlsx\"\n"
            "python -m src.train\n"
            "```"
        )
        return

    st.caption(
        " · ".join(
            f"{target_title(t)}: {len(m['id2name'])} categories"
            for t, m in bundle["targets"].items()
        )
        + f" · ensemble weights XGB {WEIGHTS['xgb']:.2f} / SVM {WEIGHTS['svm']:.2f}"
        f" / LR {WEIGHTS['lr']:.2f}"
    )

    single_tab, batch_tab, corrections_tab = st.tabs(
        ["Single TIR", "Batch file", "Corrections"]
    )
    with single_tab:
        render_single(bundle)
    with batch_tab:
        render_batch(bundle)
    with corrections_tab:
        render_corrections(bundle)


main()