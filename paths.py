#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.paths
# Author: chollida
# Created: 2026-06-29
# Last Modified: 2026-07-22
# Implements: REQ-001 (Project-wide path constants)

from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ROOT points to the repository root.
#
# The project is laid out with the modules under src/ and the settings under
# config/, but the repository is also published as a flat export with every
# file at the top level.  Resolving the root from the directory name rather
# than assuming one level up means the same code runs in both, instead of
# computing a root outside the checkout and failing to find config.json.
ROOT: Path = _HERE.parent if _HERE.name == "src" else _HERE


# Locates the settings file in whichever layout is in use.
def _config_path() -> Path:
    """Return the path to config.json, nested or flat."""
    nested = ROOT / "config" / "config.json"
    return nested if nested.is_file() else ROOT / "config.json"


CONFIG_PATH: Path = _config_path()

DATA_DIR: Path = ROOT / "data"
MODEL_DIR: Path = ROOT / "models"
REPORT_DIR: Path = ROOT / "reports"
