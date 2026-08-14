"""Offline keyword/phrase lexicon for act-tag derivation."""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Negation cues that, when directly preceding a matched pattern, suppress the
# match. Handles the "No kissing: ...", "not exposed", "without bondage"
# patterns produced by negative-space prompts without needing full NLP.
_NEGATION_WORDS = (
    "no",
    "not",
    "without",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "don't",
    "doesn't",
    "didn't",
    "never",
    "cannot",
    "can't",
    "absence",
    "lack",
    "none",
)

# How many chars before a match to scan for a trailing negation cue.
_NEGATION_WINDOW = 30

_NEGATION_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(w) for w in _NEGATION_WORDS)})\s*[:\-]?\s*$"
)


def _is_negated(text: str, match_start: int) -> bool:
    """Return True when a negation cue directly precedes ``match_start``.

    Checks the ``_NEGATION_WINDOW`` chars immediately before the match
    position for a trailing negation word (optionally followed by
    punctuation like ``:`` or ``-`` and whitespace).

    Args:
        text: The full lowered caption text.
        match_start: Start index of the pattern match within ``text``.

    Returns:
        True if the match appears to be negated.
    """
    prefix = text[max(0, match_start - _NEGATION_WINDOW) : match_start]
    return bool(_NEGATION_RE.search(prefix))


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

    Negation-aware: a match is suppressed when a negation cue (``no``,
    ``not``, ``without``, …) directly precedes it within a small window.
    This prevents false positives from negative-space prompts that list
    what is *not* happening (e.g. ``"No kissing: their faces are apart"``
    no longer triggers the ``kissing`` tag).

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
        hits: list[str] = []
        for pattern in patterns:
            lowered = str(pattern).lower()
            regex = re.compile(rf"(?<!\w){re.escape(lowered)}(?!\w)")
            for m in regex.finditer(text):
                if not _is_negated(text, m.start()):
                    hits.append(str(pattern))
                    break
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
