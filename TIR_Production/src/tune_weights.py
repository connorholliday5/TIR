#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.tune_weights
# Searches the ensemble weights for each flat target on the validation split.
#
# Run after training, from the repository root:
#     python -m src.tune_weights                 # report only
#     python -m src.tune_weights --write         # also update the ensemble.json files
#
# The search runs on the VALIDATION split only.  Tuning on test would make the
# reported figures optimistic, since the weights would have been fitted to the
# very rows used to judge them.

import argparse
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:  # modules under src/
    from src.inference import TARGETS, _score, load_bundle
    from src.paths import DATA_DIR, MODEL_DIR
    from src.utils import build_feature_matrix_many, ensemble_score_matrix, expand_proba
except ImportError:  # flat layout, modules at the repository root
    from inference import TARGETS, _score, load_bundle
    from paths import DATA_DIR, MODEL_DIR
    from utils import build_feature_matrix_many, ensemble_score_matrix, expand_proba


PROC_DIR = DATA_DIR / "processed"


# Yields every weight combination on a grid, normalised to sum to one.
def weight_grid(names, step: float = 0.05):
    """Yield weightings over `names` in `step` increments."""
    points = [round(i * step, 4) for i in range(int(1 / step) + 1)]
    for combination in product(points, repeat=len(names) - 1):
        remainder = round(1.0 - sum(combination), 4)
        if remainder < 0:
            continue
        yield dict(zip(names, (*combination, remainder)))


# Searches the grid for one target.
def tune_target(target: str, entry: dict, X, y_true) -> dict:
    """Return the best weighting for `target`, scored by macro-F1.

    Macro-F1 rather than accuracy: the categories are unbalanced enough that
    accuracy barely moves when the rare ones are all wrong, which is exactly
    the failure the weighting could help with.
    """
    num_labels = len(entry["id2name"])
    names = list(entry["models"])
    if len(names) < 2:
        print(f"\n=== {target} — only {names[0] if names else 'no'} model kept; nothing to tune ===")
        return {}

    # Scored once per model, reused for every candidate weighting.
    probas = {}
    for name, model in entry["models"].items():
        if name == "xgb":
            import xgboost as xgb_lib
            probas[name] = expand_proba(
                model.predict(xgb_lib.DMatrix(X)), np.arange(num_labels), num_labels
            )
        else:
            probas[name] = expand_proba(model.predict_proba(X), model.classes_, num_labels)

    id2name = entry["id2name"]

    def score(weights) -> float:
        combined = ensemble_score_matrix(probas, weights, num_labels)
        predicted = [id2name.get(int(i), "") for i in combined.argmax(axis=1)]
        return float(f1_score(y_true, predicted, average="macro", zero_division=0))

    scored = sorted(((score(w), w) for w in weight_grid(names)), key=lambda p: -p[0])
    best_score, best = scored[0]
    current = entry["weights"]
    current_score = score(current)

    print(f"\n=== {target} — {len(y_true):,} validation rows ===")
    print(f"  current  {current}  ->  {current_score:.4f}")
    print(f"  best     {best}  ->  {best_score:.4f}   ({best_score - current_score:+.4f})")
    for value, weights in scored[1:4]:
        print(f"    {weights}  {value:.4f}")

    return best if best_score > current_score else current


# Runs the search from the command line.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Update each ensemble.json")
    args = parser.parse_args()

    val_df = pd.read_csv(PROC_DIR / "val.csv", low_memory=False)
    bundle = load_bundle()
    X = build_feature_matrix_many(
        val_df["text"].astype(str).tolist(), bundle["tfidf_word"], bundle["tfidf_char"]
    )

    for target, spec in TARGETS.items():
        if spec.get("parent"):
            print(f"\n=== {target} — per-parent models, weights not tuned ===")
            continue

        entry = bundle["targets"][target]
        rows = val_df[f"label_{target}"].to_numpy() >= 0
        y_true = [
            entry["id2name"].get(int(i), "")
            for i in val_df.loc[rows, f"label_{target}"].astype(int)
        ]

        chosen = tune_target(target, entry, X[rows], y_true)
        if args.write and chosen:
            path = MODEL_DIR / target / "ensemble.json"
            payload = json.loads(path.read_text())
            payload["weights"] = chosen
            path.write_text(json.dumps(payload, indent=4))
            print(f"  ✔ wrote {path.relative_to(MODEL_DIR.parent)}")

    if not args.write:
        print("\n(report only — pass --write to update the ensemble.json files)")


if __name__ == "__main__":
    main()
