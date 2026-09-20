"""Taxonomy-based skill extraction (word-boundary regex over a curated skills list)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"

# A skill must not be glued to letters/digits/+/# so "Java" won't match "JavaScript"
# and "SQL" won't match "MySQL".
_BOUNDARY_L = r"(?<![A-Za-z0-9+#])"
_BOUNDARY_R = r"(?![A-Za-z0-9+#])"


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[str, str, re.Pattern]]:
    taxonomy = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    compiled = []
    for category, skills in taxonomy.items():
        for canonical, aliases in skills.items():
            names = sorted({canonical.lower(), *[a.lower() for a in aliases]}, key=len, reverse=True)
            pattern = _BOUNDARY_L + "(?:" + "|".join(re.escape(n) for n in names) + ")" + _BOUNDARY_R
            compiled.append((category, canonical, re.compile(pattern, re.IGNORECASE)))
    return compiled


def extract_skills(text: str) -> dict[str, list[str]]:
    """Return {category: [canonical skill names]} found in text."""
    flat = re.sub(r"\s+", " ", text)  # PDFs often break phrases across lines
    found: dict[str, list[str]] = {}
    for category, canonical, pattern in _compiled():
        if pattern.search(flat):
            found.setdefault(category, []).append(canonical)
    return {cat: sorted(names) for cat, names in found.items()}


def flatten(skills_by_category: dict[str, list[str]]) -> set[str]:
    return {skill for names in skills_by_category.values() for skill in names}
