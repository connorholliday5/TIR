#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.app
# Web UI for coding Technical Issue Report liabilities.
#
# Run from the repository root:
#     streamlit run src/app.py

import sys
from pathlib import Path

# `streamlit run` executes this file as a plain script rather than as part of
# the `src` package, so the repository root is not on sys.path and the imports
# below would fail with "ModuleNotFoundError: No module named 'src'".  This has
# to run before any project import.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parent, _HERE):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from datetime import datetime

import pandas as pd
import streamlit as st

try:  # modules under src/
    from src.config import (
        FEEDBACK_CFG, FEEDBACK_PATH, PRIMARY_TARGET, REQUIRED_COLS, ROOT,
        TARGETS, TEXT_COLS, heading, target_title,
    )
    from src.data import build_text_series, canonicalize, validate_input_dataframe
    from src.export import DETAIL_FIELDS, filename, to_workbook
    from src.feedback import (
        DEFAULT_SIMILARITY_THRESHOLD, append_feedback, latest_corrections, load_feedback,
    )
    from src.inference import classify, load_bundle
except ImportError:  # flat layout, modules at the repository root
    from config import (
        FEEDBACK_CFG, FEEDBACK_PATH, PRIMARY_TARGET, REQUIRED_COLS, ROOT,
        TARGETS, TEXT_COLS, heading, target_title,
    )
    from data import build_text_series, canonicalize, validate_input_dataframe
    from export import DETAIL_FIELDS, filename, to_workbook
    from feedback import (
        DEFAULT_SIMILARITY_THRESHOLD, append_feedback, latest_corrections, load_feedback,
    )
    from inference import classify, load_bundle


FEEDBACK_ENABLED = FEEDBACK_CFG.get("enabled", False)
SIMILARITY_THRESHOLD = FEEDBACK_CFG.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)

# The parent every child target hangs from; confirming it is the single
# largest accuracy lever the UI has.
ROOT_TARGET = PRIMARY_TARGET


# Loads every model artifact once per session.
@st.cache_resource(show_spinner="Loading models…")
def cached_bundle() -> dict:
    return load_bundle()


# Turns the form's fields into the text the models read.
def compose_text(fields: dict) -> str:
    """Build the model input from whichever description fields were filled."""
    frame = pd.DataFrame([{key: fields.get(key, "") for key in TEXT_COLS}])
    return build_text_series(frame, TEXT_COLS).iloc[0]


# Runs the classifier for one TIR, honouring a confirmed parent category.
def classify_one(fields: dict, bundle: dict, confirmed_parent: str = "") -> dict:
    text = compose_text(fields)
    overrides = {ROOT_TARGET: [confirmed_parent]} if confirmed_parent else None
    return classify([text], bundle, overrides).iloc[0].to_dict()


# Renders one target's prediction.
def render_prediction(target: str, result: dict) -> None:
    label = result.get(f"pred_{target}", "")
    confidence = float(result.get(f"confidence_{target}", 0.0))
    source = result.get(f"source_{target}", "model")

    left, right = st.columns([3, 1])
    left.markdown(f"**{target_title(target)}**")
    left.write(label or "—")
    right.metric("Confidence", f"{confidence:.0%}", label_visibility="collapsed")

    alternates = result.get(f"alt_{target}")
    if alternates:
        st.caption(f"Other likely codes: {alternates}")

    if source == "no parent":
        st.caption("Not predicted — the level above it has no answer.")
    elif source == "no model for parent":
        st.caption("Too few examples under this parent to model; showing its most common code.")
    elif source == "confirmed by reviewer":
        st.caption("Confirmed by you.")
    elif result.get(f"review_{target}"):
        st.caption("⚠ Below the review threshold — worth a second look.")


# Renders the single-TIR tab.
def render_single(bundle: dict) -> None:
    st.subheader("Code a single TIR")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    description = st.text_area("Description 1", height=140, key="description_1")

    with st.expander("TIR details (optional — they sharpen the prediction and fill the export)"):
        columns = st.columns(3)
        details = {
            "doc_number": columns[0].text_input(heading("doc_number"), key="f_doc_number"),
            "item_id": columns[1].text_input(heading("item_id"), key="f_item_id"),
            "item_type": columns[2].text_input(heading("item_type"), key="f_item_type"),
            "severity": columns[0].text_input(heading("severity"), key="f_severity"),
            "doc_title": columns[1].text_input(heading("doc_title"), key="f_doc_title"),
        }
        details["description_2"] = st.text_area(
            heading("description_2"), height=80, key="f_description_2"
        )

    if st.button("Classify", type="primary"):
        if not description.strip():
            st.warning("Enter a description first.")
            return
        fields = {"description_1": description, **details}
        st.session_state["fields"] = fields
        st.session_state["confirmed_parent"] = ""
        st.session_state["result"] = classify_one(fields, bundle)

    result = st.session_state.get("result")
    if not result:
        return

    fields = st.session_state.get("fields", {})

    st.divider()

    # Confirming the parent first is what makes the levels below it worth
    # trusting: routed by a category the model guessed, the sub-level scores
    # about 81%; routed by a confirmed one it scores about 90%.
    st.markdown(f"#### {target_title(ROOT_TARGET)}")
    categories = sorted(bundle["targets"][ROOT_TARGET]["id2name"].values())
    predicted = result.get(f"pred_{ROOT_TARGET}", "")
    confidence = float(result.get(f"confidence_{ROOT_TARGET}", 0.0))

    left, right = st.columns([3, 1])
    choice = left.selectbox(
        "Confirm or correct this, then the levels below are predicted inside it",
        categories,
        index=categories.index(predicted) if predicted in categories else 0,
        key="parent_choice",
    )
    right.metric("Confidence", f"{confidence:.0%}")

    if result.get(f"review_{ROOT_TARGET}"):
        st.warning("The model is unsure of this one — confirming it is worth the most here.")

    if choice != predicted and st.button("Apply and re-predict the levels below"):
        st.session_state["confirmed_parent"] = choice
        st.session_state["result"] = classify_one(fields, bundle, choice)
        if FEEDBACK_ENABLED:
            append_feedback(
                FEEDBACK_PATH, compose_text(fields), predicted, choice, ROOT_TARGET
            )
        st.rerun()

    st.divider()
    for target in bundle["order"]:
        if target == ROOT_TARGET:
            continue
        render_prediction(target, result)

    st.divider()
    if st.button("Add to session export", type="secondary"):
        record = {
            **{key: fields.get(key, "") for key in DETAIL_FIELDS},
            **{k: v for k, v in result.items()},
            "source": result.get(f"source_{ROOT_TARGET}", "model"),
            "classified_at": datetime.now().isoformat(timespec="seconds"),
        }
        rows = st.session_state.setdefault("session_rows", [])

        # Re-classifying the same TIR replaces its earlier row rather than
        # adding a second one, so the export reflects the answer a coder
        # settled on rather than every attempt along the way.
        key = (record.get("doc_number", ""), record.get("item_id", ""), record.get("description_1", ""))
        for index, existing in enumerate(rows):
            if (existing.get("doc_number", ""), existing.get("item_id", ""),
                    existing.get("description_1", "")) == key:
                rows[index] = record
                break
        else:
            rows.append(record)

        st.session_state["flash"] = f"Added — {len(rows)} TIR(s) ready to export."
        st.rerun()


# Renders the export controls shared by the single-TIR tab.
def render_session_export() -> None:
    rows = st.session_state.get("session_rows", [])

    st.subheader("Session export")
    if not rows:
        st.caption(
            "Nothing yet. Classify a TIR and choose **Add to session export**; "
            "everything you add here downloads as one Excel file."
        )
        return

    st.caption(f"{len(rows)} TIR(s) classified this session.")
    preview = pd.DataFrame([{
        heading("doc_number"): r.get("doc_number", ""),
        heading("item_id"): r.get("item_id", ""),
        heading("description_1"): (r.get("description_1", "") or "")[:70],
        **{target_title(t): r.get(f"pred_{t}", "") for t in TARGETS},
    } for r in rows])
    st.dataframe(preview, use_container_width=True, hide_index=True)

    left, right = st.columns([1, 4])
    left.download_button(
        "Download Excel",
        to_workbook(rows),
        file_name=filename(),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    if right.button("Clear session"):
        st.session_state["session_rows"] = []
        st.rerun()


# Chooses the columns worth putting in front of a reviewer.
def display_columns(out: pd.DataFrame, targets: list) -> list:
    """Return the readable subset of a predictions frame, in reading order."""
    columns = [c for c in TEXT_COLS if c in out.columns]
    for target in targets:
        columns += [f"pred_{target}", f"confidence_{target}", f"review_{target}"]
        source = f"source_{target}"
        if source in out.columns and (out[source] != "model").any():
            columns.append(source)
    return [c for c in columns if c in out.columns]


# Renders the batch-upload tab.
def render_batch(bundle: dict) -> None:
    st.subheader("Classify a file")
    st.caption(
        "CSV or Excel from QPS. Any of the export layouts is accepted — the "
        "column names are recognised automatically."
    )

    upload = st.file_uploader("TIR export", type=["csv", "xlsx", "xls"])
    if upload is None:
        return

    raw = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    df = canonicalize(raw)

    try:
        validate_input_dataframe(df, REQUIRED_COLS)
    except ValueError as exc:
        st.error(str(exc))
        return

    texts = build_text_series(df, TEXT_COLS)
    with st.spinner(f"Classifying {len(df):,} rows…"):
        preds = classify(texts.tolist(), bundle)

    out = pd.concat([df.reset_index(drop=True), preds], axis=1)
    targets = list(bundle["order"])

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
    only_review = left.toggle("Only rows needing review")
    show_everything = right.toggle("Show every column")

    # Rows the top level is unsure of come first: fixing those is worth more
    # than fixing anything below them, since every level under a wrong parent
    # is wrong with it.
    view = out.loc[preds[f"review_{ROOT_TARGET}"]] if only_review else out
    if view.empty:
        st.info("Nothing was flagged for review — every row cleared its threshold.")
    else:
        st.dataframe(
            view if show_everything else view[display_columns(view, targets)],
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download predictions.csv",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


# Renders the saved-corrections tab.
def render_corrections() -> None:
    st.subheader("Reviewer corrections")

    log = load_feedback(FEEDBACK_PATH)
    if log.empty:
        st.info(
            "No corrections yet. Correct a category on the **Single TIR** tab "
            "and it will appear here."
        )
        return

    active = latest_corrections(log)
    st.caption(
        f"{len(log)} correction(s) logged, {len(active)} distinct wording(s) in effect. "
        f"Matching is exact, or above {SIMILARITY_THRESHOLD:.0%} similarity."
    )
    st.dataframe(log.iloc[::-1], use_container_width=True)
    st.download_button(
        "Download feedback.csv",
        log.to_csv(index=False).encode("utf-8"),
        file_name="feedback.csv",
        mime="text/csv",
    )
    st.markdown("Retrain to fold these into the models:\n```\npython -m src.train\n```")


# Builds the page.
def main() -> None:
    st.set_page_config(page_title="TIR Liability Coding", page_icon="🔧", layout="wide")
    st.title("TIR Liability Coding")

    try:
        bundle = cached_bundle()
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
            f"{target_title(t)}: {len(bundle['targets'][t]['id2name'])} categories"
            for t in bundle["order"]
        )
    )

    single_tab, batch_tab, corrections_tab = st.tabs(
        ["Single TIR", "Batch file", "Corrections"]
    )
    with single_tab:
        render_single(bundle)
        st.divider()
        render_session_export()
    with batch_tab:
        render_batch(bundle)
    with corrections_tab:
        render_corrections()


main()
