"""Filesystem layout. ``JSPACE_WORK`` redirects lenses/ and results/ (used by
the smoke test); ``JSPACE_CORPUS`` points at a JSON list of corpus records."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("JSPACE_WORK", ROOT)
DATA = os.path.join(ROOT, "data")
LENSES = os.path.join(WORK, "lenses")
RESULTS = os.path.join(WORK, "results")
CORPUS = os.environ.get("JSPACE_CORPUS", os.path.join(DATA, "wikitext103_records.json"))


def resolve(path: str) -> str:
    """Absolute path; relative paths are tried under WORK then ROOT."""
    if os.path.isabs(path):
        return path
    for base in (WORK, ROOT):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    return os.path.join(WORK, path)
