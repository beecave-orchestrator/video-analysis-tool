"""Atomic JSON sidecar read/write helpers."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sidecar_path(video_path: Path) -> Path:
    """Return ``<video>.nsfw.json`` for the given video path."""
    return video_path.with_suffix(".nsfw.json")


def read_sidecar(path: Path) -> dict[str, Any]:
    """Read and parse a sidecar JSON file.

    Args:
        path: Sidecar file path.

    Returns:
        Parsed sidecar payload.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_sidecar(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically using a temporary file + rename.

    Args:
        path: Destination sidecar path.
        data: Dictionary to serialise as JSON.
    """
    data.setdefault("schema_version", 1)
    data.setdefault("created_at", datetime.now(timezone.utc).astimezone().isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
