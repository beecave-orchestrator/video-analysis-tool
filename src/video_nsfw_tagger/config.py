"""Default configuration and constants for the video-nsfw-tagger CLI."""

from pathlib import Path

DEFAULT_VIT_MODEL = "Falconsai/nsfw_image_detection"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_THRESHOLD = 0.7
DEFAULT_FPS = 1.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_DB = "video_nsfw_index.db"
DEFAULT_LEXICON = (
    Path(__file__).resolve().parents[2] / "lexicon" / "acts.yaml"
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
