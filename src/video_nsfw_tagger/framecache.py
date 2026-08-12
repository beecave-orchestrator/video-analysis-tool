"""On-disk cache for extracted frames and ViT scores.

Caching lets repeated VLM experiments (e.g. prompt sweeps) skip frame
extraction and ViT classification entirely: the cache key covers the
video content (path + size + mtime) and sampling parameters (fps,
max_duration, ViT model), while the NSFW *threshold* is applied at load
time so it can vary between runs without invalidating the cache.
"""

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from video_nsfw_tagger.ollama import FrameRef

SCORES_FILE = "scores.json"
FRAMES_DIR = "frames"


def cache_key(
    video_path: Path,
    fps: float,
    max_duration: float | None,
    vit_model: str,
) -> str:
    """Return a stable cache key for a video and sampling configuration.

    Args:
        video_path: Video file (must exist).
        fps: Frame sampling rate used during extraction.
        max_duration: Optional cap on seconds processed.
        vit_model: ViT model name used for classification.

    Returns:
        Hex digest identifying the cache entry.
    """
    stat = video_path.resolve().stat()
    parts = "|".join([
        str(video_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        str(fps),
        str(max_duration),
        vit_model,
    ])
    return hashlib.sha1(parts.encode("utf-8")).hexdigest()[:16]


def _entry_dir(cache_dir: Path, key: str) -> Path:
    return cache_dir / key


def load(cache_dir: Path, key: str) -> tuple[list[FrameRef], list[float]] | None:
    """Load cached frames and scores.

    Args:
        cache_dir: Root frame-cache directory.
        key: Cache key from :func:`cache_key`.

    Returns:
        ``(frames, scores)`` where frames are ``(index, timestamp_s, path)``
        tuples pointing into the cache, or ``None`` on a cache miss or
        corrupt entry.
    """
    entry = _entry_dir(cache_dir, key)
    scores_path = entry / SCORES_FILE
    if not scores_path.is_file():
        return None
    try:
        data = json.loads(scores_path.read_text(encoding="utf-8"))
        frames: list[FrameRef] = []
        scores: list[float] = []
        for item in data["frames"]:
            path = entry / FRAMES_DIR / item["file"]
            if not path.is_file():
                return None
            frames.append((item["frame"], item["timestamp_s"], path))
            scores.append(item["score"])
        return frames, scores
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return None


def save(
    cache_dir: Path,
    key: str,
    frames: Sequence[FrameRef],
    scores: Sequence[float],
    metadata: dict | None = None,
) -> Path:
    """Persist extracted frames and their ViT scores.

    Args:
        cache_dir: Root frame-cache directory.
        key: Cache key from :func:`cache_key`.
        frames: ``(index, timestamp_s, path)`` tuples; frame images are
            copied into the cache.
        scores: Per-frame NSFW scores, parallel to ``frames``.
        metadata: Optional extra metadata stored alongside.

    Returns:
        Path to the cache entry directory.

    Raises:
        ValueError: If ``frames`` and ``scores`` differ in length.
    """
    if len(frames) != len(scores):
        raise ValueError(
            f"frames ({len(frames)}) and scores ({len(scores)}) length mismatch"
        )
    entry = _entry_dir(cache_dir, key)
    frames_dir = entry / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for (idx, ts, path), score in zip(frames, scores, strict=True):
        dest = frames_dir / path.name
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)
        items.append({
            "frame": idx,
            "timestamp_s": ts,
            "score": score,
            "file": path.name,
        })

    payload = {"metadata": metadata or {}, "frames": items}
    (entry / SCORES_FILE).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return entry
