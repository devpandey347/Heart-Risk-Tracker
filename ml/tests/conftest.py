# conftest.py — pytest configuration for ml/tests/
# Adds ml/src/ to sys.path so tests can import source modules directly.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
