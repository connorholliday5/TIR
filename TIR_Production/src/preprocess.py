#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.preprocess
# Performs raw-data ingestion, column canonicalisation, text cleaning,
# category normalisation, hierarchy extraction and dataset splitting.

import json
import argparse
from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
from sklearn.model_selection import train_test_split

try:  # modules under src/
    from src.columns import canonicalize, resolve_columns
    from src.paths import CONFIG_PATH, DATA_DIR, ROOT
    from src.utils import build_text_series, normalize_categories, validate_input_dataframe
except ImportError:  # flat layout, modules at the repository root
    from columns import canonicalize, resolve_columns
    from paths import CONFIG_PATH, DATA_DIR, ROOT
    from utils import build_text_series, normalize_categories, validate_input_dataframe

# Loads configuration values such as the random seed.
CONFIG = json.loads(CONFIG_PATH.read_text())
SEED = CONFIG.get("random_seed", 42)
ALIASES: dict = CONFIG.get("column_aliases", {})
REQUIRED_COLS = CONFIG.get("required_columns", ["description_1"])
TEXT_COLS = CONFIG.get("text_columns", REQUIRED_COLS)

# Each entry names a column to predict and how to tidy its values.  Adding a
# target here is all that preprocess and train need to pick it up.
TARGETS: dict = CONFIG.get("targets", {})
PRIMARY_TARGET = CONFIG.get("primary_target", next(iter(TARGETS), ""))

# Defines paths used for raw and processed datasets.
RAW_DIR: Path = DATA_DIR / "raw"
PROC_DIR: Path = DATA_DIR / "processed"


# Returns the label-map path for a target.
def label_map_path(target: str) -> Path:
    """Path of the id -> category-name map written for `target`."""
    return DATA_DIR / f"label_map_{target}.json"


# Path of the parent -> children table shared by training and inference.
HIERARCHY_PATH: Path = DATA_DIR / "hierarchy.json"


# Loads the normalisation table a target refers to, if any.
def normalization_for(spec: dict) -> dict:
    """Return the lower-case -> standard-form map named by `spec`.

    A target without a `normalization` key keeps its column values verbatim,
    which is right for a taxonomy that is already consistent.  The Process
    sub- and third-level codes are such a taxonomy: every value in the smaller
    export also appears in the larger one, spelled identically.
    """
    key = spec.get("normalization")
    if not key:
        return {}

    table = CONFIG.get(key)
    if not table:
        raise ValueError(
            f"{CONFIG_PATH.name} is missing '{key}': it maps each lower-case "
            f"value of the '{spec['column']}' column to its standard form."
        )
    return table


# Reads one raw export and renames its columns to canonical form.
def load_raw(path: Path) -> pd.DataFrame:
    """Load an Excel or CSV export with its columns canonicalised.

    Each file is canonicalised on its own, before anything is concatenated:
    the three QPS layouts name the same field three different ways
    ("Description 1", "DESCRIPTION_ONE", "Item Description 1"), so combining
    them first would produce a frame of disjoint, mostly-empty columns.
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    resolved = resolve_columns(df, ALIASES)
    print(f"  ↳ {len(df):,} rows, {len(resolved)}/{len(ALIASES)} known fields recognised")

    unrecognised = [name for name in ALIASES if name not in resolved]
    if unrecognised:
        print(f"    (absent from this file: {', '.join(unrecognised)})")

    return canonicalize(df, ALIASES)


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
    original = cast(pd.Series, df[column])
    raw = original.astype(str).str.strip()

    # An empty cell means "nobody coded this", which is not the same as "coded
    # as something the table has not seen".  Both used to end up in the
    # unknown bucket, and because a third of the larger pull is uncoded that
    # turned 30,000 blanks into a fabricated OTHER category holding a third of
    # the training rows.  Blanks are identified before normalisation and
    # withheld below, so only genuinely unrecognised values become OTHER.
    #
    # Tested with `isna` as well as against the placeholder spellings: pandas 2
    # renders a missing value as the literal string "nan" under `astype(str)`,
    # while pandas 3 keeps it missing.  Checking only one of the two silently
    # does nothing on the other version.
    blank = original.isna() | raw.fillna("").str.strip().str.lower().isin(
        ["", "nan", "none", "nat", "<na>"]
    )

    values = normalize_categories(raw, table, spec.get("unknown_value", "OTHER"))

    # Names the categories the table does not cover.  Without this they are
    # folded into the unknown bucket without comment, which is how four real
    # categories went unnoticed in the sample export.  A new file is the most
    # likely place for a spelling the table has never seen.
    if table:
        unmapped = raw[~raw.str.lower().isin(table) & ~blank].value_counts()
        if len(unmapped):
            listed = ", ".join(f"{name} ({n})" for name, n in unmapped.head(10).items())
            more = "" if len(unmapped) <= 10 else f", and {len(unmapped) - 10} more"
            print(
                f"  ⚠ {target}: {len(unmapped)} value(s) are not in "
                f"'{spec['normalization']}' and become "
                f"{spec.get('unknown_value', 'OTHER')}: {listed}{more}"
            )

    # Blank cells survive .astype(str) as the strings above; treat them as absent.
    values = values.mask(blank | values.fillna("").str.lower().isin(["", "nan", "none"]))

    counts = cast(pd.Series, values.value_counts())
    min_size = int(spec.get("min_class_size", 3))
    # str() so the join below is unambiguous: the index holds category names.
    too_rare = sorted(str(name) for name in counts.loc[counts < min_size].index)
    if too_rare:
        shown = ", ".join(too_rare[:8]) + ("" if len(too_rare) <= 8 else ", …")
        print(
            f"  ⚠ {target}: dropping {len(too_rare)} class(es) under "
            f"{min_size} records: {shown}"
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


# Records which child categories were observed under each parent.
def build_hierarchy(df: pd.DataFrame, label_maps: Dict[str, dict]) -> dict:
    """Return {target: {parent category: [child categories]}}.

    Built from the categories actually seen together in the training split
    rather than from the shape of the codes.  The codes do nest — a sub-code
    starts with its parent's code, as HAPI does under HA — but only in about
    99.9% of records at the second level and 95.5% at the third, so a string
    rule would quietly mis-file the remainder.  What the coders actually
    recorded is the more reliable source.

    Only the training split is passed in, so the validation and test rows
    cannot influence which combinations the model is allowed to predict.
    """
    hierarchy: Dict[str, Dict[str, List[str]]] = {}

    for target, spec in TARGETS.items():
        parent = spec.get("parent")
        if not parent:
            continue

        child_names = {int(k): v["name"] for k, v in label_maps[target].items()}
        parent_names = {int(k): v["name"] for k, v in label_maps[parent].items()}

        usable = df[(df[f"label_{target}"] >= 0) & (df[f"label_{parent}"] >= 0)]
        pairs = (
            usable[[f"label_{parent}", f"label_{target}"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        table: Dict[str, List[str]] = {}
        for parent_id, child_id in pairs:
            table.setdefault(parent_names[int(parent_id)], []).append(
                child_names[int(child_id)]
            )

        hierarchy[target] = {p: sorted(set(c)) for p, c in sorted(table.items())}

        widths = [len(c) for c in hierarchy[target].values()]
        print(
            f"  ✔ {target}: {len(hierarchy[target])} parent(s), "
            f"{min(widths)}-{max(widths)} children each "
            f"(median {sorted(widths)[len(widths) // 2]})"
            if widths else f"  ⚠ {target}: no parent/child pairs observed"
        )

    return hierarchy


# Runs the preprocessing pipeline from the command line.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_csv",
        nargs="+",
        required=True,
        help="Paths to one or more raw Excel or CSV files (data/raw, or anywhere)",
    )
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not TARGETS:
        raise ValueError(
            f"{CONFIG_PATH.name} is missing 'targets': it names the column to "
            "predict for each classification field."
        )
    if PRIMARY_TARGET not in TARGETS:
        raise ValueError(
            f"{CONFIG_PATH.name} sets primary_target='{PRIMARY_TARGET}', which "
            f"is not one of: {', '.join(TARGETS)}"
        )

    # A child target must be listed after its parent, so that inference can
    # predict the parent first.  Checked here rather than at inference time,
    # where the failure would be a confusing KeyError per row.
    seen: List[str] = []
    for target, spec in TARGETS.items():
        parent = spec.get("parent")
        if parent and parent not in seen:
            raise ValueError(
                f"{CONFIG_PATH.name}: target '{target}' declares parent "
                f"'{parent}', which must be defined before it in 'targets'."
            )
        seen.append(target)

    # Loads and concatenates raw data files.
    frames = []
    for file in args.raw_csv:
        # Looked for as given, then in data/raw, then beside the project and
        # one level above it — the QPS exports are normally kept next to the
        # project folder rather than inside it.
        candidates = [
            Path(file),
            RAW_DIR / Path(file).name,
            ROOT / Path(file).name,
            ROOT.parent / Path(file).name,
        ]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"Raw data file not found: {file}. Looked in: "
                + ", ".join(str(c.parent) for c in candidates)
            )

        print(f"📄 Loading {path.name}")
        frames.append(load_raw(path))

    merged = pd.concat(frames, ignore_index=True)
    print(f"✔ Merged {len(frames)} file(s). Total rows: {len(merged):,}")

    label_columns = [spec["column"] for spec in TARGETS.values()]
    validate_input_dataframe(merged, REQUIRED_COLS)

    missing_labels = [c for c in label_columns if c not in merged.columns]
    if missing_labels:
        raise ValueError(
            "No input file carries the label column(s): "
            f"{', '.join(missing_labels)}. Training needs a labelled export."
        )

    # Builds the model input text before de-duplicating, so that duplicates are
    # judged on what the model actually reads.
    merged["text"] = build_text_series(merged, TEXT_COLS)

    # Drops records repeated across files.  The sample export turned out to be
    # a 99.4% subset of the larger pull, and because the two use different
    # column names a whole-row comparison found nothing in common — every one
    # of those TIRs would have been trained on twice and split across train and
    # test.  Matching on the canonical text and labels is what actually catches
    # it: two rows the model reads identically and that carry the same answers
    # cannot teach it anything different.
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text", *label_columns], keep="first")
    merged = merged.reset_index(drop=True)
    if len(merged) < before:
        print(f"✔ Removed {before - len(merged):,} duplicate row(s). Remaining: {len(merged):,}")

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

    # Records the parent/child taxonomy from the training split only.
    print("Extracting hierarchy:")
    label_maps = {
        t: json.loads(label_map_path(t).read_text()) for t in TARGETS
    }
    HIERARCHY_PATH.write_text(json.dumps(build_hierarchy(train_df, label_maps), indent=4))
    print(f"✔ Saved hierarchy: {HIERARCHY_PATH}")

    # No embedding corpus is exported: the models read TF-IDF features that
    # src.train fits directly from the text.

    print("\n Preprocessing complete!\n")


if __name__ == "__main__":
    main()
