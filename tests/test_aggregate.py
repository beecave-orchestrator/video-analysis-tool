"""Tests for the aggregate module."""

import pytest

from video_nsfw_tagger.aggregate import aggregate


def test_aggregate_all_normal():
    result = aggregate([0.1, 0.2, 0.3], threshold=0.7, fps=1)
    assert result.verdict == "normal"
    assert result.max_score == 0.3
    assert result.nsfw_percent == 0.0
    assert not result.flagged_frames


def test_aggregate_one_flag():
    result = aggregate([0.1, 0.85, 0.2], threshold=0.7, fps=1)
    assert result.verdict == "nsfw"
    assert result.max_score == 0.85
    assert result.nsfw_percent == pytest.approx(33.33, rel=1e-2)
    assert result.flagged_frames == [{"frame": 2, "timestamp_s": 1.0, "score": 0.85}]


def test_aggregate_empty():
    result = aggregate([], threshold=0.7, fps=1)
    assert result.verdict == "normal"
    assert result.max_score == 0.0
    assert result.nsfw_percent == 0.0
