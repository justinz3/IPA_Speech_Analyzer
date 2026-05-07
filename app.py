"""Hugging Face Spaces entry point.

HF Spaces looks for a top-level `app.py` (configured via README frontmatter).
This is just a thin shim into `vocal_ipa.web.build_app`; the real UI lives
in the package so it stays in sync with the CLI.
"""

from vocal_ipa.web import build_app

app = build_app()

if __name__ == "__main__":
    app.launch()
