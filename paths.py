#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.paths
# Author: chollida
# Created: 2026-06-29
# Last Modified: 2026-07-22
# Implements: REQ-001 (Project‑wide path constants)

from pathlib import Path

# ROOT points to the repository root (one level above src/)
ROOT: Path = Path(__file__).resolve().parent.parent