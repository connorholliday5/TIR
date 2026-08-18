#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.tune_weights
# Searches the ensemble weights on the validation split.
#
# The three model outputs are blended with fixed weights from config.json.
# Those were chosen by hand; this grid-searches them against held-out data and
# prints the best set, so the choice is measured rather than assumed.
#
# The search runs on the VALIDATION split only.  Tuning on test would make the
# reported accuracy optimistic, since the weights would have been fitted to
# the very rows used to judge them.
#
# Run after training, from the repository root:
#     python -m src.tune_weights                 # report only
#     python -m src.tune_weights --write         # also update config.json

import argparse
import json
from itertools import product

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.embedding import SubwordHashEmbedder
from src.paths import ROOT
from src.utils import (
    build_feature_matrix,
    clean_text,
    combine_ensemble_scores,
    expand_proba,
    vstack_csr,
)


CONFIG_PATH = ROOT / "config" / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
TARGETS: dict = CONFIG.get("targets", {})

PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


# Yields every weight combination on a grid, normalised to sum to one.
def weight_grid(step: float = 0.05):
    """Yield {"xgb","svm","lr"} weightings in `step` increments."""
    points = [round(i * step, 4) for i in range(int(1 / step) + 1)]
    for xgb_w, svm_w in product(points, points):
        lr_w = round(1.0 - xgb_w - svm_w, 4)
        if lr_w < 0:
            continue
        yield {"xgb": xgb_w, "svm": svm_w, "lr": lr_w}


# Scores one weighting against the validation labels.
def accuracy_for(weights, svm_pred, lr_proba, xgb_proba, y_true, num_labels) -> float:
    """Return validation accuracy for a candidate weighting."""
    pred, _ = combine_ensemble_scores(svm_pred, lr_proba, xgb_proba, weights, num_labels)
    return float((pred == y_true).mean())


# Searches the grid for one target.
def tune_target(target: str, val_df: pd.DataFrame, features) -> dict:
    """Return the best weighting for `target` and how it compares to config."""
    raw = json.loads((DATA_DIR / f"label_map_{target}.json").read_text())
    num_labels = len(raw)

    rows = val_df[f"label_{target}"].to_numpy() >= 0
    y_true = val_df.loc[rows, f"label_{target}"].astype(int).to_numpy()
    X = features[rows]

    target_dir = MODEL_DIR / target
    svm_model = joblib.load(target_dir / "svm.pkl")
    lr_model = joblib.load(target_dir / "lr.pkl")
    booster = xgb.Booster()
    booster.load_model(str(target_dir / "xgb.json"))

    # Scored once, reused for every candidate weighting.
    svm_pred = svm_model.predict(X)
    lr_proba = expand_proba(lr_model.predict_proba(X), lr_model.classes_, num_labels)
    xgb_proba = booster.predict(xgb.DMatrix(X))

    current = CONFIG.get("ensemble_weights", {"xgb": 0.40, "svm": 0.40, "lr": 0.20})
    scored = [
        (accuracy_for(w, svm_pred, lr_proba, xgb_proba, y_true, num_labels), w)
        for w in weight_grid()
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    best_acc, best = scored[0]
    current_acc = accuracy_for(current, svm_pred, lr_proba, xgb_proba, y_true, num_labels)

    print(f"\n=== {target} — {len(y_true):,} validation rows ===")
    print(f"  current  xgb {current['xgb']:.2f} / svm {current['svm']:.2f} / "
          f"lr {current['lr']:.2f}  ->  {current_acc:.4f}")
    print(f"  best     xgb {best['xgb']:.2f} / svm {best['svm']:.2f} / "
          f"lr {best['lr']:.2f}  ->  {best_acc:.4f}   ({best_acc - current_acc:+.4f})")
    print("  runners-up:")
    for acc, w in scored[1:4]:
        print(f"    xgb {w['xgb']:.2f} / svm {w['svm']:.2f} / lr {w['lr']:.2f}  {acc:.4f}")

    return {"weights": best, "accuracy": best_acc, "current_accuracy": current_acc}


# Runs the search from the command line.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the best weighting back into config/config.json",
    )
    args = parser.parse_args()

    val_df = pd.read_csv(PROC_DIR / "val.csv")
    val_df["text"] = val_df["text"].astype(str).apply(clean_text)

    tfidf_word = joblib.load(MODEL_DIR / "tfidf_word.pkl")
    tfidf_char = joblib.load(MODEL_DIR / "tfidf_char.pkl")
    embedder = SubwordHashEmbedder.load(MODEL_DIR / "dense_meta.json")

    features = vstack_csr([
        build_feature_matrix(t, tfidf_word, tfidf_char, embedder) for t in val_df["text"]
    ])

    results = {t: tune_target(t, val_df, features) for t in TARGETS}

    # One weighting is shared by every target, so pick the set with the best
    # mean validation accuracy rather than letting the last target win.
    candidates = {}
    for result in results.values():
        key = tuple(sorted(result["weights"].items()))
        candidates[key] = result["weights"]

    if len(candidates) == 1:
        chosen = next(iter(candidates.values()))
        print(f"\nEvery target agrees on xgb {chosen['xgb']:.2f} / "
              f"svm {chosen['svm']:.2f} / lr {chosen['lr']:.2f}")
    else:
        print("\nTargets prefer different weightings; keeping the configured set.")
        print("Re-run with a single target, or accept the small per-target loss.")
        chosen = CONFIG.get("ensemble_weights")

    if args.write and chosen:
        CONFIG["ensemble_weights"] = chosen
        CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2) + "\n")
        print(f"✔ Wrote ensemble_weights to {CONFIG_PATH.relative_to(ROOT)}")
    elif not args.write:
        print("\n(report only — pass --write to update config.json)")


if __name__ == "__main__":
    main()