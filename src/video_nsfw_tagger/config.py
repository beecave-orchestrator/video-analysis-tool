"""Default configuration and constants for the video-nsfw-tagger CLI."""

from pathlib import Path

DEFAULT_VIT_MODEL = "Falconsai/nsfw_image_detection"
DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_THRESHOLD = 0.7
DEFAULT_FPS = 1.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_DB = "video_nsfw_index.db"
DEFAULT_LEXICON = (
    Path(__file__).resolve().parents[3] / "lexicon" / "acts.yaml"
)

SUPPORTED_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".wmv",
}
