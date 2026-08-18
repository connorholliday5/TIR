#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.preprocess
# Performs raw-data ingestion, text cleaning, category normalisation, and dataset splitting.

import json
import argparse
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import ROOT
from src.utils import build_text_series, normalize_categories, validate_input_dataframe

# Loads configuration values such as the random seed.
CONFIG = json.loads((ROOT / "config" / "config.json").read_text())
SEED = CONFIG.get("random_seed", 42)
REQUIRED_COLS = CONFIG.get("required_columns", ["Description 1"])

# Each entry names a column to predict and how to tidy its values.  Adding a
# target here is all that preprocess and train need to pick it up.
TARGETS: dict = CONFIG.get("targets", {})
PRIMARY_TARGET = CONFIG.get("primary_target", next(iter(TARGETS), ""))

# Defines paths used for raw and processed datasets.
RAW_DIR: Path = ROOT / "data" / "raw"
PROC_DIR: Path = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR: Path = ROOT / "data"


# Returns the label-map path for a target.
def label_map_path(target: str) -> Path:
    """Path of the id -> category-name map written for `target`."""
    return DATA_DIR / f"label_map_{target}.json"


# Loads the normalisation table a target refers to, if any.
def normalization_for(spec: dict) -> dict:
    """Return the lower-case -> standard-form map named by `spec`.

    A target without a `normalization` key keeps its column values verbatim,
    which is right for a taxonomy that is already consistent.
    """
    key = spec.get("normalization")
    if not key:
        return {}

    table = CONFIG.get(key)
    if not table:
        raise ValueError(
            f"config/config.json is missing '{key}': it maps each lower-case "
            f"value of the '{spec['column']}' column to its standard form."
        )
    return table


# Standardises one target column and assigns it numeric labels.
def prepare_target(df: pd.DataFrame, target: str, spec: dict) -> pd.Series:
    """Normalise `spec['column']`, write its label map, and return the labels.

    Rows whose category is blank, or belongs to a class with fewer than
    `min_class_size` records, are labelled -1 and skipped when this target is
    trained.  They stay in the dataset because another target may still be
    usable on the same row.

    Returns:
        An integer Series of label ids, -1 where the row is unusable.
    """
    column = spec["column"]
    table = normalization_for(spec)
    raw = cast(pd.Series, df[column]).astype(str).str.strip()
    values = normalize_categories(raw, table, spec.get("unknown_value", "OTHER"))

    # Names the categories the table does not cover.  Without this they are
    # folded into the unknown bucket without comment, which is how four real
    # categories went unnoticed in the sample export.  A new file is the most
    # likely place for a spelling the table has never seen.
    if table:
        blank = raw.str.lower().isin(["", "nan", "none"])
        unmapped = raw[~raw.str.lower().isin(table) & ~blank].value_counts()
        if len(unmapped):
            listed = ", ".join(f"{name} ({n})" for name, n in unmapped.head(10).items())
            more = "" if len(unmapped) <= 10 else f", and {len(unmapped) - 10} more"
            print(
                f"  ⚠ {target}: {len(unmapped)} value(s) are not in "
                f"'{spec['normalization']}' and become "
                f"{spec.get('unknown_value', 'OTHER')}: {listed}{more}"
            )

    # Blank cells survive .astype(str) as the strings below; treat them as absent.
    values = values.mask(values.str.lower().isin(["", "nan", "none"]))

    counts = cast(pd.Series, values.value_counts())
    min_size = int(spec.get("min_class_size", 3))
    # str() so the join below is unambiguous: the index holds category names.
    too_rare = sorted(str(name) for name in counts.loc[counts < min_size].index)
    if too_rare:
        print(
            f"  ⚠ {target}: dropping {len(too_rare)} class(es) under "
            f"{min_size} records: {', '.join(too_rare)}"
        )
        values = values.mask(values.isin(too_rare))

    categories = sorted(values.dropna().unique())
    label_map = {i: {"name": name} for i, name in enumerate(categories)}
    label_map_path(target).write_text(json.dumps(label_map, indent=4))

    name2id = {name: i for i, name in enumerate(categories)}
    labels = values.map(name2id).fillna(-1).astype(int)

    usable = int((labels >= 0).sum())
    print(f"  ✔ {target}: {len(categories)} categories, {usable:,} usable rows "
          f"({usable / len(df):.1%})")
    return labels


# Runs the preprocessing pipeline from the command line.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_csv",
        nargs="+",
        required=True,
        help="Paths to one or more raw Excel or CSV files under data/raw",
    )
    args = parser.parse_args()

    if not TARGETS:
        raise ValueError(
            "config/config.json is missing 'targets': it names the column to "
            "predict for each classification field."
        )
    if PRIMARY_TARGET not in TARGETS:
        raise ValueError(
            f"config/config.json sets primary_target='{PRIMARY_TARGET}', which "
            f"is not one of: {', '.join(TARGETS)}"
        )

    # Loads and concatenates raw data files.
    frames = []
    for file in args.raw_csv:
        path = RAW_DIR / Path(file).name
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")

        print(f"📄 Loading {path}")

        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {file}")

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"✔ Merged {len(frames)} files. Total rows: {len(merged)}")

    label_columns = [spec["column"] for spec in TARGETS.values()]
    validate_input_dataframe(merged, [*REQUIRED_COLS, *label_columns])

    # Drops records repeated across files.  The two sample exports are byte
    # identical, so passing both would otherwise duplicate every record and put
    # the same TIR into the training and the test split.
    #
    # Matching is on the whole row, not just the description: two distinct TIRs
    # can legitimately share a description and labels while differing in serial
    # number or date, and those are real records that belong in training.
    before = len(merged)
    merged = merged.drop_duplicates(keep="first")
    merged = merged.reset_index(drop=True)
    if len(merged) < before:
        print(f"✔ Removed {before - len(merged):,} duplicate row(s). Remaining: {len(merged):,}")

    # Builds the model input text.  Checked up front rather than defaulted
    # with .get(): a missing column made .get() return a plain str, so the
    # next .astype() failed with an unrelated AttributeError.
    merged["text"] = build_text_series(merged, REQUIRED_COLS)

    # Normalises each target column and assigns its numeric labels.
    print("Preparing targets:")
    for target, spec in TARGETS.items():
        merged[f"label_{target}"] = prepare_target(merged, target, spec)

    # Saves the cleaned training dataset.
    cleaned_path = PROC_DIR / "cleaned_for_training.csv"
    merged.to_csv(cleaned_path, index=False)
    print(f"✔ Saved cleaned data: {cleaned_path}")

    # Splits the dataset once, stratified on the primary target so its rarer
    # categories stay represented.  Rows the primary target cannot use are
    # still split, since another target may be able to use them.
    primary_col = f"label_{PRIMARY_TARGET}"
    stratify = merged[primary_col] if merged[primary_col].min() >= 0 else None

    # train_test_split is annotated as returning a list; it returns one frame
    # per input, so the pair is named explicitly.
    train_df, test_df = cast(
        "tuple[pd.DataFrame, pd.DataFrame]",
        train_test_split(merged, test_size=0.15, random_state=SEED, stratify=stratify),
    )
    train_df, val_df = cast(
        "tuple[pd.DataFrame, pd.DataFrame]",
        train_test_split(
            train_df, test_size=0.15, random_state=SEED,
            stratify=train_df[primary_col] if stratify is not None else None,
        ),
    )

    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    val_df.to_csv(PROC_DIR / "val.csv", index=False)
    test_df.to_csv(PROC_DIR / "test.csv", index=False)
    print(f"✔ Saved train.csv ({len(train_df):,}), val.csv ({len(val_df):,}), "
          f"test.csv ({len(test_df):,})")

    # No embedding corpus is exported: the dense subword embeddings are
    # derived from the text itself by src.train, so there is nothing to fit.

    print("\n Preprocessing complete!\n")


if __name__ == "__main__":
    main()