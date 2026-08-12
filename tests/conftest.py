"""Shared test fixtures."""

import subprocess

import pytest


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """Generate a synthetic 5-second test video with ffmpeg.

    Returns:
        Path to the generated video file.
    """
    tmp = tmp_path_factory.mktemp("fixtures")
    video = tmp / "sample.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=5:size=320x240:rate=30",
        "-pix_fmt",
        "yuv420p",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return video
