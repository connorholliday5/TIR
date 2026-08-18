#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.config
# Project paths and the settings every other module reads.
#
# config.json used to be parsed independently by preprocessing, training,
# inference and the reports, each deriving the same handful of values from it
# in slightly different ways.  Loading it once here means a setting cannot mean
# one thing during training and another at prediction time.

import json
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent

# ROOT points to the project root.
#
# Resolved from the directory name rather than assumed to be one level up, so
# the same code runs whether the modules sit under src/ or are checked out flat
# at the top level — a flat export would otherwise compute a root outside the
# checkout and fail to find config.json.
ROOT: Path = _HERE.parent if _HERE.name == "src" else _HERE


# Locates the settings file in whichever layout is in use.
def _config_path() -> Path:
    nested = ROOT / "config" / "config.json"
    return nested if nested.is_file() else ROOT / "config.json"


CONFIG_PATH: Path = _config_path()
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROC_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = ROOT / "models"
REPORT_DIR: Path = ROOT / "reports"

FEEDBACK_PATH: Path = DATA_DIR / "feedback.csv"
HIERARCHY_PATH: Path = DATA_DIR / "hierarchy.json"

CONFIG: dict = json.loads(CONFIG_PATH.read_text())

SEED: int = CONFIG.get("random_seed", 42)
ALIASES: Dict[str, List[str]] = CONFIG.get("column_aliases", {})
REQUIRED_COLS: List[str] = CONFIG.get("required_columns", ["description_1"])
TEXT_COLS: List[str] = CONFIG.get("text_columns", REQUIRED_COLS)
TARGETS: dict = CONFIG.get("targets", {})
PRIMARY_TARGET: str = CONFIG.get("primary_target", next(iter(TARGETS), ""))

GATE: dict = CONFIG.get("gate", {})
FEEDBACK_CFG: dict = CONFIG.get("feedback", {})

DEFAULT_REVIEW_THRESHOLD: float = CONFIG.get("review_threshold", 0.69)


# Returns the label-map path for a target.
def label_map_path(target: str) -> Path:
    """Path of the id -> category-name map written for `target`."""
    return DATA_DIR / f"label_map_{target}.json"


# Returns the review threshold configured for one target.
def review_threshold(target: str) -> float:
    """Confidence below which `target` is flagged for a human.

    Set per target: the levels differ enormously in how consistently people
    code them, so one global number would either wave through the deepest
    level or flag almost everything at the top.
    """
    return float(TARGETS.get(target, {}).get("review_threshold", DEFAULT_REVIEW_THRESHOLD))


# Human-facing name for a target.
def target_title(target: str) -> str:
    """Readable label, e.g. "process_cat" -> "Process Cat".

    Taken from the first alias of the target's column, so the name a coder
    sees is the heading the TIR team's own exports use.
    """
    column = TARGETS.get(target, {}).get("column", target)
    names = ALIASES.get(column) or ALIASES.get(target)
    return names[0] if names else target.replace("_", " ").title()


# Gives a canonical field the heading the TIR team reads it under.
def heading(canonical: str) -> str:
    """Return the export heading for a canonical field name."""
    names = ALIASES.get(canonical)
    return names[0] if names else canonical.replace("_", " ").title()


# Checks that a child target is defined after the parent it depends on.
def validate_target_order() -> None:
    """Raise if a target names a parent that is not defined before it.

    Inference predicts parents first, so a child listed above its parent would
    fail once per row with a confusing KeyError.  Caught here instead.
    """
    seen: List[str] = []
    for target, spec in TARGETS.items():
        parent = spec.get("parent")
        if parent and parent not in seen:
            raise ValueError(
                f"{CONFIG_PATH.name}: target '{target}' declares parent "
                f"'{parent}', which must be defined before it in 'targets'."
            )
        seen.append(target)
