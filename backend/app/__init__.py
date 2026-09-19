"""App package bootstrap.

Make the vendored ``minuspod_compat`` package importable at runtime without a
build step: the venv has no pip and compat's declared ``requests`` dep is only
needed by the (lazily imported) pricing surface the proxy never touches.
"""

from __future__ import annotations

import sys
from pathlib import Path

_COMPAT_DIR = Path(__file__).resolve().parents[2] / "compat"
if _COMPAT_DIR.is_dir() and str(_COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPAT_DIR))
