"""Offline keyword/phrase lexicon for act-tag derivation."""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_lexicon(path: Path) -> dict[str, Any]:
    """Load a lexicon from JSON or YAML.

    Args:
        path: Lexicon file path.

    Returns:
        Mapping of tag names to lists of keyword/phrase patterns.

    Raises:
        ImportError: If a YAML lexicon is given but PyYAML isn't installed.
        ValueError: If the file extension isn't ``.json``/``.yaml``/``.yml``.
    """
    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if ext == ".json":
        return json.loads(text)
    if ext in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML lexicons; "
                "install it or use a JSON lexicon."
            ) from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported lexicon format: {path}")


def find_matches(
    caption: str, lexicon: dict[str, Iterable[str]]
) -> dict[str, list[str]]:
    """Return the matched patterns per tag for ``caption``.

    Matching is case-insensitive and whole-phrase: a pattern only matches
    when it is not embedded inside a larger word (e.g. ``ass`` does not
    match ``glasses``), so common substrings don't produce false positives.

    Args:
        caption: VLM-generated caption text.
        lexicon: ``{tag: [patterns]}`` mapping.

    Returns:
        ``{tag: [matched patterns]}`` mapping; tags with no matches are
        omitted.
    """
    text = caption.lower()
    matches: dict[str, list[str]] = {}
    for tag, patterns in lexicon.items():
        hits = [
            str(pattern)
            for pattern in patterns
            if re.search(rf"(?<!\w){re.escape(str(pattern).lower())}(?!\w)", text)
        ]
        if hits:
            matches[tag] = hits
    return matches


def find_tags(caption: str, lexicon: dict[str, Iterable[str]]) -> list[str]:
    """Return the sorted tags whose patterns appear in ``caption``.

    Thin wrapper over :func:`find_matches` when only tag names are needed.

    Args:
        caption: VLM-generated caption text.
        lexicon: ``{tag: [patterns]}`` mapping.

    Returns:
        Sorted, unique list of matched tags.
    """
    return sorted(find_matches(caption, lexicon))
