"""Aggregate per-frame NSFW scores into a video-level result."""

from dataclasses import dataclass, field


@dataclass
class AggregateResult:
    """Per-video NSFW aggregation."""

    nsfw_percent: float
    max_score: float
    verdict: str
    flagged_frames: list[dict] = field(default_factory=list)


def aggregate(
    scores: list[float],
    threshold: float,
    fps: float = 1.0,
) -> AggregateResult:
    """Aggregate frame scores into video statistics.

    Args:
        scores: Per-frame NSFW scores.
        threshold: Score threshold for flagging a frame.
        fps: Frames per second used during extraction (for timestamps).

    Returns:
        Populated ``AggregateResult``.
    """
    total = len(scores)
    flagged: list[dict] = []
    max_score = 0.0

    for i, score in enumerate(scores):
        max_score = max(max_score, score)
        if score >= threshold:
            flagged.append({
                "frame": i + 1,
                "timestamp_s": round(i / fps, 3),
                "score": round(score, 4),
            })

    percent = (len(flagged) / total * 100.0) if total else 0.0
    verdict = "nsfw" if max_score >= threshold else "normal"

    return AggregateResult(
        nsfw_percent=round(percent, 2),
        max_score=round(max_score, 4),
        verdict=verdict,
        flagged_frames=flagged,
    )
