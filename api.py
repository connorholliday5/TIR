#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.api
# Implements a deterministic dense‑embedding inference API.

import json

import joblib
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

from src.embedding import SubwordHashEmbedder
from src.paths import ROOT
from src.utils import (
    build_feature_matrix,
    clean_text,
    combine_ensemble_scores,
    expand_proba,
    is_blank_text,
    verify_model_hash,
)


# Loads configuration and ensemble weights.
CONFIG = json.loads((ROOT / "config" / "config.json").read_text())
WEIGHTS = CONFIG.get("ensemble_weights", {"xgb": 0.40, "svm": 0.40, "lr": 0.20})
REVIEW_THRESHOLD = CONFIG.get("review_threshold", 0.55)
TARGETS = CONFIG.get("targets", {})

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


# Verifies integrity before loading: joblib.load unpickles, so the check is
# only meaningful while the files are still untouched bytes on disk.
for _name in ("tfidf_word.pkl", "tfidf_char.pkl", "dense_meta.json"):
    verify_model_hash(MODEL_DIR / _name)

tfidf_word = joblib.load(MODEL_DIR / "tfidf_word.pkl")
tfidf_char = joblib.load(MODEL_DIR / "tfidf_char.pkl")

# Rebuilds the exact embedder that training used.
embedder = SubwordHashEmbedder.load(MODEL_DIR / "dense_meta.json")


# Loads the classifiers and label map for every configured target.
def _load_targets() -> dict:
    loaded = {}
    for target in TARGETS:
        target_dir = MODEL_DIR / target
        for name in ("svm.pkl", "lr.pkl", "xgb.json"):
            verify_model_hash(target_dir / name)

        booster = xgb.Booster()
        booster.load_model(str(target_dir / "xgb.json"))

        raw = json.loads((DATA_DIR / f"label_map_{target}.json").read_text())
        loaded[target] = {
            "svm": joblib.load(target_dir / "svm.pkl"),
            "lr": joblib.load(target_dir / "lr.pkl"),
            "xgb": booster,
            "id2name": {int(k): v["name"] for k, v in raw.items()},
        }
    return loaded


MODELS = _load_targets()


# Creates the FastAPI application.
app = FastAPI(title="TIR Liability API – Deterministic Dense Embedding")


# Describes the JSON body a caller posts to /predict.
class TIRInput(BaseModel):
    # Matches the single description column the models are trained on.
    description: str


# Builds the full feature vector using TF-IDF and dense embeddings.
def build_features(text: str):
    return build_feature_matrix(text, tfidf_word, tfidf_char, embedder)


# Runs ensemble prediction for every configured target.
@app.post("/predict")
def predict(item: TIRInput):
    text = clean_text(item.description)

    # No description, no answer.  The models would still score an all-zero
    # vector and can report high confidence doing it, so nothing is guessed.
    if is_blank_text(text):
        return {
            "text": text,
            "predictions": {
                target: {"id": None, "label": "", "confidence": 0.0, "review_flag": True}
                for target in MODELS
            },
            "note": "description was empty; no classification attempted",
        }

    X = build_features(text)

    predictions = {}
    for target, bundle in MODELS.items():
        num_labels = len(bundle["id2name"])

        ids, confidences = combine_ensemble_scores(
            bundle["svm"].predict(X),
            expand_proba(bundle["lr"].predict_proba(X), bundle["lr"].classes_, num_labels),
            bundle["xgb"].predict(xgb.DMatrix(X)),
            WEIGHTS,
            num_labels,
        )

        label_id = int(ids[0])
        confidence = float(confidences[0])
        predictions[target] = {
            "id": label_id,
            "label": bundle["id2name"].get(label_id, ""),
            "confidence": confidence,
            "review_flag": confidence < REVIEW_THRESHOLD,
        }

    return {"text": text, "predictions": predictions}