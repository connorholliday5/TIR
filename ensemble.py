#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.ensemble
# Provides deterministic dense embedding ensemble inference.

import argparse
import json
from pathlib import Path
from typing import cast

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

from src.embedding import SubwordHashEmbedder
from src.paths import ROOT
from src.utils import (
    build_feature_matrix,
    build_text_series,
    combine_ensemble_scores,
    expand_proba,
    is_blank_text,
    normalize_categories,
    validate_input_dataframe,
    verify_model_hash,
    vstack_csr,
)


# Loads the id -> category-name map for one target.
def load_label_map(path: Path) -> dict:
    """Return {label id: category name} for a target."""
    raw = json.loads(path.read_text())
    return {int(k): v["name"] for k, v in raw.items()}


# Loads the classifiers for one target.
def load_target_models(model_dir: Path, target: str) -> dict:
    """Verify and load the SVM, LR and XGBoost models for `target`."""
    target_dir = model_dir / target

    # Integrity first: joblib.load unpickles, so the check is only meaningful
    # while the files are still untouched bytes on disk.
    for name in ("svm.pkl", "lr.pkl", "xgb.json"):
        verify_model_hash(target_dir / name)

    booster = xgb.Booster()
    booster.load_model(str(target_dir / "xgb.json"))
    return {
        "svm": joblib.load(target_dir / "svm.pkl"),
        "lr": joblib.load(target_dir / "lr.pkl"),
        "xgb": booster,
    }


# Runs deterministic ensemble inference on a CSV or Excel input file.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True, help="CSV or Excel file to classify")
    parser.add_argument("--out", required=True, help="Where to write the predictions CSV")
    parser.add_argument("--models_dir", default=str(ROOT / "models"))
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    args = parser.parse_args()

    # Loads configuration, ensemble weights and the targets to predict.
    CONFIG = json.loads((ROOT / "config" / "config.json").read_text())
    WEIGHTS = CONFIG.get("ensemble_weights", {"xgb": 0.40, "svm": 0.40, "lr": 0.20})
    REQUIRED_COLS = CONFIG.get("required_columns", ["Description 1"])
    REVIEW_THRESHOLD = CONFIG.get("review_threshold", 0.55)
    TARGETS = CONFIG.get("targets", {})

    model_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)

    # Loads the input dataset.
    fp = Path(args.raw_csv)
    df = pd.read_csv(fp) if fp.suffix.lower() == ".csv" else pd.read_excel(fp)

    validate_input_dataframe(df, REQUIRED_COLS)
    df["text"] = build_text_series(df, REQUIRED_COLS)

    # Loads the shared vectorizers and embedder, verifying them first.
    for name in ("tfidf_word.pkl", "tfidf_char.pkl", "dense_meta.json"):
        verify_model_hash(model_dir / name)

    tfidf_word = joblib.load(model_dir / "tfidf_word.pkl")
    tfidf_char = joblib.load(model_dir / "tfidf_char.pkl")
    embedder = SubwordHashEmbedder.load(model_dir / "dense_meta.json")

    # Builds feature vectors once; every target scores the same matrix.
    X = vstack_csr([
        build_feature_matrix(txt, tfidf_word, tfidf_char, embedder) for txt in df["text"]
    ])

    for target, spec in TARGETS.items():
        id2name = load_label_map(data_dir / f"label_map_{target}.json")
        num_labels = len(id2name)
        models = load_target_models(model_dir, target)

        pred, conf = combine_ensemble_scores(
            models["svm"].predict(X),
            expand_proba(models["lr"].predict_proba(X), models["lr"].classes_, num_labels),
            models["xgb"].predict(xgb.DMatrix(X)),
            WEIGHTS,
            num_labels,
        )

        df[f"pred_{target}"] = [id2name.get(int(i), "") for i in pred]
        df[f"confidence_{target}"] = conf
        df[f"review_{target}"] = conf < REVIEW_THRESHOLD

        # A row with no description gets no answer.  The models still score an
        # all-zero vector and can report high confidence doing it, so the
        # result is withheld and the row sent for review.
        blank = df["text"].map(is_blank_text).to_numpy()
        if blank.any():
            df.loc[blank, f"pred_{target}"] = ""
            df.loc[blank, f"confidence_{target}"] = 0.0
            df.loc[blank, f"review_{target}"] = True

        # Reports metrics when the input carries the true category.  The truth
        # column holds raw values, so it goes through the same normalisation
        # preprocessing applied before the labels were built.
        column = spec["column"]
        if column in df.columns:
            table = CONFIG.get(spec.get("normalization", ""), {})
            truth = normalize_categories(
                cast(pd.Series, df[column]), table, spec.get("unknown_value", "")
            )
            known = truth.isin(set(id2name.values()))
            if known.any():
                acc = accuracy_score(truth[known], df.loc[known, f"pred_{target}"])
                print(f"\n=== {target} ({column}) — {int(known.sum()):,} labelled rows ===")
                print(f"Accuracy: {acc:.6f}\n")
                # zero_division=0 reports a category the models never predicted
                # as 0.0 rather than warning.  sklearn documents 0 as valid but
                # annotates the parameter as str, hence the ignore.
                print(classification_report(
                    truth[known], df.loc[known, f"pred_{target}"], digits=4,
                    zero_division=0,  # type: ignore[arg-type]
                ))
            else:
                print(f"\n⚠ {target}: no rows carry a known '{column}' value; metrics skipped.")

    # Writes the output CSV.
    Path(args.out).write_text(df.to_csv(index=False))
    print("✔ Saved predictions:", args.out)


if __name__ == "__main__":
    main()