"""Tests for the frame + ViT score cache."""

import pytest

from video_nsfw_tagger import framecache


def test_cache_key_is_stable_and_unique(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")
    key1 = framecache.cache_key(video, 1.0, None, "Falconsai/nsfw_image_detection")
    key2 = framecache.cache_key(video, 1.0, None, "Falconsai/nsfw_image_detection")
    assert key1 == key2
    assert len(key1) == 16

    key3 = framecache.cache_key(video, 2.0, None, "Falconsai/nsfw_image_detection")
    assert key3 != key1


def test_save_and_load_roundtrip(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")
    frames = [
        (1, 0.0, tmp_path / "frame_0001.png"),
        (2, 1.0, tmp_path / "frame_0002.png"),
    ]
    for _, _, p in frames:
        p.write_bytes(b"img")
    scores = [0.1, 0.9]

    cache_dir = tmp_path / "cache"
    key = framecache.cache_key(video, 1.0, None, "Falconsai/nsfw_image_detection")
    entry = framecache.save(
        cache_dir,
        key,
        frames,
        scores,
        metadata={"video_path": str(video)},
    )

    assert entry.is_dir()
    assert (entry / "frames" / "frame_0001.png").exists()

    loaded = framecache.load(cache_dir, key)
    assert loaded is not None
    loaded_frames, loaded_scores = loaded
    assert [f[0] for f in loaded_frames] == [1, 2]
    assert loaded_scores == scores
    assert loaded_frames[0][2].is_file()


def test_load_returns_none_on_missing(tmp_path):
    assert framecache.load(tmp_path, "nonexistent") is None


def test_save_checks_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        framecache.save(tmp_path, "k", [(1, 0.0, tmp_path / "x")], [0.1, 0.9])
