#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.config
# Implements:
#   - REQ-001 (Project-wide path constants)
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

# A branch is a family of codes a record either carries or does not carry at
# all.  Process codes are a TIR field and Program codes are a SURV field: in
# the three-year pull 54,340 records have Process but no Program, 15,740 have
# Program but no Process, and only 7,733 have both.  The in-branch models only
# ever saw coded rows, so asked about a record outside their branch they answer
# confidently and wrongly.  Each branch therefore gets its own classifier
# deciding whether the branch applies before any code within it is predicted.
BRANCHES: dict = CONFIG.get("branches", {})

# What each code means, supplied by the coding team.  Optional: the models
# learn from examples and need none of it.  A language model is the opposite —
# reading the definition is how it can reach a code with two examples behind
# it — so this file is what decides whether consulting one is worth doing.
CODE_DEFINITIONS: Path = CONFIG_PATH.parent / "code_definitions.json"

# Settings for the language model consulted where the classifiers are weakest.
LLM: dict = CONFIG.get("llm", {})

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


# Name of the label column holding whether one branch applies to a record.
def branch_label(branch: str) -> str:
    """Column name preprocessing writes the branch-applicability label under."""
    return f"branch_{branch}"


# Human-facing name for a branch.
def branch_title(branch: str) -> str:
    """Readable label for a branch, e.g. "program" -> "Program codes"."""
    return BRANCHES.get(branch, {}).get("title", f"{branch.title()} codes")


# Returns the confidence a branch must reach before its codes are filled in.
def branch_threshold(branch: str) -> float:
    """Confidence below which the whole branch is left to a human."""
    return float(BRANCHES.get(branch, {}).get("review_threshold", DEFAULT_REVIEW_THRESHOLD))


# Groups the targets into the rows the QPS entry screen shows.
def families() -> List[List[str]]:
    """Return each chain of targets, deepest-first within the chain.

    QPS presents the codes as a grid — Metric, Process and Program across
    Category, Sub-Category and 3rd Level — and a coder reading the app should
    see the screen they already know.  The grid is not configured separately
    because it is already implied by which target names which parent: a family
    is a root target and everything descending from it.

    A flat target with no children (DST) comes back as a chain of one.
    """
    children: Dict[str, List[str]] = {}
    roots: List[str] = []
    for target, spec in TARGETS.items():
        parent = spec.get("parent")
        if parent:
            children.setdefault(parent, []).append(target)
        else:
            roots.append(target)

    def chain(target: str) -> List[str]:
        line = [target]
        while children.get(line[-1]):
            line.append(children[line[-1]][0])
        return line

    return [chain(root) for root in roots]


# Checks that a child target is defined after the parent it depends on.
def validate_target_order() -> None:
    """Raise if a target names a parent or branch that is not defined first.

    Inference predicts parents first, so a child listed above its parent would
    fail once per row with a confusing KeyError.  Caught here instead.  The
    same applies to a branch: it decides whether its targets are answered at
    all, so it has to exist before anything claims membership of it.
    """
    seen: List[str] = []
    for target, spec in TARGETS.items():
        parent = spec.get("parent")
        if parent and parent not in seen:
            raise ValueError(
                f"{CONFIG_PATH.name}: target '{target}' declares parent "
                f"'{parent}', which must be defined before it in 'targets'."
            )

        branch = spec.get("branch")
        if branch and branch not in BRANCHES:
            raise ValueError(
                f"{CONFIG_PATH.name}: target '{target}' declares branch "
                f"'{branch}', which is not defined in 'branches'."
            )
        seen.append(target)

    for branch, spec in BRANCHES.items():
        anchor = spec.get("anchor")
        if anchor not in TARGETS:
            raise ValueError(
                f"{CONFIG_PATH.name}: branch '{branch}' is anchored on "
                f"'{anchor}', which is not a target."
            )
