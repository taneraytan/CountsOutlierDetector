"""Shared test configuration.

The database path env var is set before any app module is imported so tests
can never touch a real user database in ``~/.counts_outlier_detector``.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "COUNTS_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="cod_tests_"), "data.db"),
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
