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


def test_compute_metrics_includes_score(tmp_path):
    """Score = distinct_tags_count * tagged_frames_pct / 100."""
    run_dir = tmp_path / "run"
    p1 = run_dir / "01_a"
    p1.mkdir(parents=True)
    p1.joinpath("v.nsfw.json").write_text(
        json.dumps(
            _make_sidecar(
                "01_a",
                [
                    {"frame": 1, "caption": "A naked person."},
                    {"frame": 2, "caption": "An intimate scene."},
                ],
            )
        )
    )
    rows = prompt_report.collect(run_dir, LEXICON)
    # 2 tags (nudity + intimacy), 100% tagged → score = 2.0
    assert rows[0]["score"] == 2.0


def test_rank_sorts_by_score_desc(tmp_path):
    """Higher score ranks first; 100% tagged with more tags beats 50% with more."""
    run_dir = tmp_path / "run"
    p_reliable = run_dir / "01_reliable"
    p_reliable.mkdir(parents=True)
    p_reliable.joinpath("v.nsfw.json").write_text(
        json.dumps(
            _make_sidecar(
                "01_reliable",
                [
                    {"frame": 1, "caption": "A naked intimate scene with a vibrator."},
                    {"frame": 2, "caption": "Another naked intimate scene."},
                ],
            )
        )
    )
    p_flaky = run_dir / "02_flaky"
    p_flaky.mkdir(parents=True)
    p_flaky.joinpath("v.nsfw.json").write_text(
        json.dumps(
            _make_sidecar(
                "02_flaky",
                [
                    {"frame": 1, "caption": "A naked intimate scene with a vibrator."},
                    {"frame": 2, "caption": ""},
                ],
            )
        )
    )
    rows = prompt_report.collect(run_dir, LEXICON)
    ranked = prompt_report._rank(rows)
    # reliable: 3 tags × 100% = 3.0; flaky: 3 tags × 50% = 1.5
    assert ranked[0]["prompt_id"] == "01_reliable"
    assert ranked[0]["score"] == 3.0
    assert ranked[1]["prompt_id"] == "02_flaky"
    assert ranked[1]["score"] == 1.5


def test_markdown_report_includes_score_column(tmp_path):
    """The markdown summary table has a score column."""
    run_dir = tmp_path / "run"
    p1 = run_dir / "01_a"
    p1.mkdir(parents=True)
    p1.joinpath("v.nsfw.json").write_text(
        json.dumps(
            _make_sidecar(
                "01_a", [{"frame": 1, "timestamp_s": 1.0, "caption": "A naked person."}]
            )
        )
    )
    rows = prompt_report.collect(run_dir, LEXICON)
    md = prompt_report._markdown_report(rows, LEXICON)
    assert "score" in md
    assert "| score |" in md
