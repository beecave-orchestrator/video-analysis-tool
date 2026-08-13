"""Default configuration and constants for the video-nsfw-tagger CLI."""

import logging
import os
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

DEFAULT_VIT_MODEL = "Falconsai/nsfw_image_detection_26"

# Path to a .env file that may hold an HF_TOKEN for gated models. The 2026
# Falconsai ViT is gated and requires a Hugging Face read token to download.
# We reuse the token already configured for the insanely-fast-whisper-rocm
# stack rather than asking the user to log in again.
HF_TOKEN_ENV_PATH = Path.home() / ".config" / "insanely-fast-whisper-rocm" / ".env"


def load_hf_token() -> str | None:
    """Resolve a Hugging Face access token without exposing the secret.

    Precedence:
      1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variables.
      2. ``HF_TOKEN`` defined in :data:`HF_TOKEN_ENV_PATH` (read via
         ``python-dotenv`` without mutating the process environment).

    Returns:
        The token string, or ``None`` if no token is configured. The returned
        value must never be logged.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var, "").strip()
        if value:
            return value

    env_path = HF_TOKEN_ENV_PATH
    if not env_path.is_file():
        return None

    try:
        values = dotenv_values(env_path)
    except OSError as exc:
        logger.debug("Could not read HF token file %s: %s", env_path, exc)
        return None

    value = (values.get("HF_TOKEN") or "").strip()
    return value or None


DEFAULT_OLLAMA_MODEL = (
    "hf.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M"
)
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_THRESHOLD = 0.7
DEFAULT_FPS = 1.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_DB = "video_nsfw_index.db"
DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "lexicon" / "acts.yaml"

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
