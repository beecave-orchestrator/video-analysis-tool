"""Tests for prompt comparison report aggregation."""

import json

from video_nsfw_tagger import prompt_report

LEXICON = {
    "nudity": ["naked", "nude"],
    "intimacy": ["intimate"],
    "toys_objects": ["vibrator", "dildo"],
}


def _make_sidecar(prompt_id: str, captions: list) -> dict:
    return {
        "video_path": "/tmp/video.mp4",
        "vlm": {
            "prompt_id": prompt_id,
            "captions": captions,
        },
    }


def test_collect_computes_expected_metrics(tmp_path):
    run_dir = tmp_path / "run"
    p1_dir = run_dir / "01_baseline"
    p1_dir.mkdir(parents=True)
    sidecar = p1_dir / "video.mp4.nsfw.json"
    sidecar.write_text(
        json.dumps(
            _make_sidecar(
                "01_baseline",
                [
                    {"frame": 1, "caption": "A naked person with a vibrator."},
                    {"frame": 2, "caption": "An intimate scene."},
                ],
            )
        )
    )

    rows = prompt_report.collect(run_dir, LEXICON)
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt_id"] == "01_baseline"
    assert row["succeeded"] == 2
    assert row["distinct_tags_count"] == 3
    assert row["tagged_frames"] == 2
    assert row["pattern_counts"]["nudity"] == 1
    assert row["pattern_counts"]["toys_objects"] == 1
    assert row["pattern_counts"]["intimacy"] == 1


def test_collect_handles_json_caption_output(tmp_path):
    run_dir = tmp_path / "run"
    pdir = run_dir / "07_json"
    pdir.mkdir(parents=True)
    pdir.joinpath("video.mp4.nsfw.json").write_text(
        json.dumps(
            _make_sidecar(
                "07_json",
                [
                    {
                        "frame": 1,
                        "caption": {
                            "description": "A naked person.",
                            "acts": ["fucking"],
                            "objects": ["vibrator"],
                        },
                    },
                ],
            )
        )
    )

    rows = prompt_report.collect(
        run_dir, {"nudity": ["naked"], "toys_objects": ["vibrator"]}
    )
    assert rows[0]["distinct_tags_count"] == 2
