"""Shared test config.

Adds the project root to sys.path so `import main`, `import schemas`, etc.
work without an installed package. Keeps the test suite zero-friction —
just `.venv/bin/pytest tests/` from the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
