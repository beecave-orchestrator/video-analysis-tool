"""ffmpeg extraction integration tests."""

import pytest

from video_nsfw_tagger import extract


@pytest.mark.slow
@pytest.mark.integration
def test_extract_duration(synthetic_video):
    dur = extract.get_duration(synthetic_video)
    assert dur == pytest.approx(5.0, abs=0.1)


@pytest.mark.slow
@pytest.mark.integration
def test_extract_frames(synthetic_video):
    with extract.extracted_frames(synthetic_video, fps=1) as frames:
        assert len(frames) == 5
        _, timestamp, _ = frames[0]
        assert timestamp == pytest.approx(0.0, abs=0.01)
