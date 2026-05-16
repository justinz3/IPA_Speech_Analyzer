"""Hugging Face Spaces entry point.

HF Spaces looks for a top-level `app.py` (configured via README frontmatter).
This is just a thin shim into `vocal_ipa.web.build_app`; the real UI lives
in the package so it stays in sync with the CLI.
"""

import sys
from pathlib import Path

# src/ layout: make vocal_ipa importable when run directly (not installed).
_src = Path(__file__).parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from importlib.resources import files

from vocal_ipa.web import build_app

_data_dir = str(files("vocal_ipa") / "data")
app = build_app()

if __name__ == "__main__":
    app.launch(allowed_paths=[_data_dir])
