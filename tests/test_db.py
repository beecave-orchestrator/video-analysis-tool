"""Tests for the SQLite index module."""

from video_nsfw_tagger.db import init_db, query_videos, upsert_video


def test_upsert_and_query(tmp_path):
    db_path = tmp_path / "index.db"
    conn = init_db(db_path)
    record = {
        "path": str(tmp_path / "a.mp4"),
        "duration_s": 5.0,
        "frames_total": 5,
        "nsfw_percent": 0.0,
        "max_score": 0.1,
        "verdict": "normal",
        "threshold": 0.7,
        "model": "Falconsai/nsfw_image_detection",
        "vlm_model": None,
        "act_tags": "[]",
        "sidecar_path": str(tmp_path / "a.nsfw.json"),
        "scanned_at": "2026-01-01T00:00:00",
    }
    upsert_video(conn, record)
    rows = query_videos(conn)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "normal"

    record["verdict"] = "nsfw"
    upsert_video(conn, record)
    rows = query_videos(conn, verdict="nsfw")
    assert len(rows) == 1
    assert rows[0]["verdict"] == "nsfw"
