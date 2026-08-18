#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.api
# Exposes the classifier over HTTP.

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

try:  # modules under src/
    from src.config import TEXT_COLS
    from src.data import build_text_series, clean_text, is_blank_text
    from src.inference import classify, load_bundle
except ImportError:  # flat layout, modules at the repository root
    from config import TEXT_COLS
    from data import build_text_series, clean_text, is_blank_text
    from inference import classify, load_bundle

import pandas as pd


BUNDLE = load_bundle()

app = FastAPI(title="TIR Liability Coding API")


# Describes the JSON body a caller posts to /predict.
class TIRInput(BaseModel):
    # The description columns the models are trained on.  Only the first is
    # required; the others sharpen the answer where a caller has them, which
    # is worth about five points of macro-F1 on the rarer categories.
    description: str
    description_2: str = ""
    doc_title: str = ""
    # A confirmed Process Category, when the caller already knows it.  Supplying
    # it lifts the sub-category by roughly nine points, because the child
    # classifiers are then chosen by a known parent rather than a guessed one.
    process_cat: Optional[str] = None


# Runs prediction for every configured target.
@app.post("/predict")
def predict(item: TIRInput):
    frame = pd.DataFrame([{
        "description_1": item.description,
        "description_2": item.description_2,
        "doc_title": item.doc_title,
    }])
    text = build_text_series(frame, TEXT_COLS).iloc[0]

    # No description, no answer.  The models would still score an all-zero
    # vector and can report high confidence doing it, so nothing is guessed.
    if is_blank_text(text):
        return {
            "text": text,
            "predictions": {
                target: {"label": "", "confidence": 0.0, "review_flag": True}
                for target in BUNDLE["order"]
            },
            "note": "description was empty; no classification attempted",
        }

    overrides = {"process_cat": [item.process_cat]} if item.process_cat else None
    row = classify([clean_text(text)], BUNDLE, overrides).iloc[0]

    predictions = {}
    for target in BUNDLE["order"]:
        entry = {
            "label": row[f"pred_{target}"],
            "confidence": float(row[f"confidence_{target}"]),
            "review_flag": bool(row[f"review_{target}"]),
            "source": row[f"source_{target}"],
        }
        alternates = row.get(f"alt_{target}")
        if alternates:
            entry["alternates"] = [a for a in str(alternates).split(", ") if a]
        predictions[target] = entry

    return {"text": text, "predictions": predictions}
