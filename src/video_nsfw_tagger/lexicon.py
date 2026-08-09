"""Offline keyword/phrase lexicon for act-tag derivation."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_lexicon(path: Path) -> Dict[str, Any]:
    """Load a lexicon from JSON or YAML.

    Args:
        path: Lexicon file path.

    Returns:
        Mapping of tag names to lists of keyword/phrase patterns.
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


def find_tags(caption: str, lexicon: Dict[str, Iterable[str]]) -> List[str]:
    """Return the sorted tags whose patterns appear in ``caption``.

    Matching is case-insensitive and exact-substring for the whole phrase.

    Args:
        caption: VLM-generated caption text.
        lexicon: ``{tag: [patterns]}`` mapping.

    Returns:
        Sorted, unique list of matched tags.
    """
    text = caption.lower()
    matched = set()
    for tag, patterns in lexicon.items():
        for pattern in patterns:
            if str(pattern).lower() in text:
                matched.add(tag)
                break
    return sorted(matched)
