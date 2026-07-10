"""Pytest bootstrap.

Ensures the project root (this directory) is on ``sys.path`` so tests can
``import app`` regardless of where pytest is invoked from. Kept at the project
root so its directory is added to ``sys.path`` during collection.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
