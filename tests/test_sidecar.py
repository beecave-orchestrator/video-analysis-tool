"""Tests for the sidecar module."""

from pathlib import Path

from video_nsfw_tagger.sidecar import read_sidecar, sidecar_path, write_sidecar


def test_sidecar_path():
    assert sidecar_path(Path("clip.mp4")) == Path("clip.nsfw.json")


def test_sidecar_roundtrip(tmp_path):
    video = tmp_path / "movie.avi"
    side = sidecar_path(video)
    data = {"video_path": str(video), "verdict": "normal", "max_score": 0.2}
    write_sidecar(side, data)
    loaded = read_sidecar(side)
    assert loaded["verdict"] == "normal"
    assert loaded["schema_version"] == 1
    assert "created_at" in loaded
