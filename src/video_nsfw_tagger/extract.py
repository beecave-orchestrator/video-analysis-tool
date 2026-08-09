"""ffmpeg / ffprobe frame extraction helpers."""

import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple


def get_duration(video_path: Path) -> float:
    """Return the video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    out = result.stdout.strip()
    if not out:
        raise RuntimeError(f"ffprobe returned no duration for {video_path}")
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration {out!r}") from exc


def _extract_to_dir(
    video_path: Path,
    fps: float,
    max_duration: float | None,
    tmp: Path,
) -> List[Tuple[int, float, Path]]:
    """Run ffmpeg and collect sorted frame paths with their timestamps."""
    time_args = ["-t", str(max_duration)] if max_duration is not None else []
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        *time_args,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(tmp / "frame_%04d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr.strip()}")

    pattern = re.compile(r"^frame_(\d{4})\.png$")
    frames: List[Tuple[int, float, Path]] = []
    for frame_path in sorted(tmp.iterdir()):
        match = pattern.match(frame_path.name)
        if match:
            idx = int(match.group(1))
            timestamp = (idx - 1) / fps
            frames.append((idx, timestamp, frame_path))
    return frames


@contextmanager
def extracted_frames(
    video_path: Path,
    fps: float = 1.0,
    max_duration: float | None = None,
) -> Iterator[List[Tuple[int, float, Path]]]:
    """Yield extracted frames from a temporary directory and clean it up on exit."""
    tmp = Path(tempfile.mkdtemp(prefix="vnt_frames_"))
    try:
        yield _extract_to_dir(video_path, fps, max_duration, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
