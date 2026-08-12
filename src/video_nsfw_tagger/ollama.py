"""Ollama HTTP API captioner — first concrete VLM backend.

Calls a local Ollama server (default ``http://localhost:11434``) with a
vision model (default ``config.DEFAULT_OLLAMA_MODEL``, an abliterated
Qwen3-VL 4B) to caption NSFW-flagged frames.
``vlm.py`` remains a placeholder for a future transformers-based backend.
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import ollama

from video_nsfw_tagger import config

logger = logging.getLogger(__name__)

# Targeted description prompt — produces richer captions that match more
# lexicon patterns than a generic "describe this image".
DEFAULT_PROMPT = (
    "Describe what is happening in this image: the people, their actions, "
    "state of dress, and the setting."
)

# Generous default: the "Thinking" VLM variants occasionally produce very
# long reasoning; a cold model load also counts against the first request.
REQUEST_TIMEOUT_S = 300
KEEP_ALIVE = "10m"

# (frame_index, timestamp_s, path)
FrameRef = tuple[int, float, Path]


def select_flagged_frames(
    frames: Sequence[FrameRef],
    flagged_frames: Sequence[dict],
    top_k: int | None = None,
) -> list[FrameRef]:
    """Select flagged frames for captioning, capped by score.

    Args:
        frames: All extracted frames as ``(index, timestamp_s, path)``.
        flagged_frames: Aggregation output dicts with ``frame`` and ``score``.
        top_k: Optional cap; highest-scoring frames are kept.

    Returns:
        Frame refs to caption, ordered by descending score.
    """
    by_index = {idx: (idx, ts, path) for idx, ts, path in frames}
    selected = sorted(flagged_frames, key=lambda f: f["score"], reverse=True)
    if top_k is not None:
        selected = selected[:top_k]
    return [by_index[f["frame"]] for f in selected if f["frame"] in by_index]


class OllamaCaptioner:
    """Caption frames via the Ollama HTTP API."""

    def __init__(
        self,
        model: str = config.DEFAULT_OLLAMA_MODEL,
        host: str = config.DEFAULT_OLLAMA_HOST,
        prompt: str = DEFAULT_PROMPT,
        timeout: float = REQUEST_TIMEOUT_S,
        keep_alive: str = KEEP_ALIVE,
    ) -> None:
        """Initialise the captioner with an Ollama client.

        Args:
            model: Ollama model name (e.g. ``qwen3-vl:4b``).
            host: Ollama server URL.
            prompt: Captioning prompt sent with each image.
            timeout: Per-request timeout in seconds.
            keep_alive: How long Ollama keeps the model loaded per request.
        """
        self.model = model
        self.host = host
        self.prompt = prompt
        self.keep_alive = keep_alive
        self.client = ollama.Client(host=host, timeout=timeout)

    def check_available(self) -> None:
        """Fail fast if Ollama is unreachable or the model isn't pulled.

        Raises:
            RuntimeError: If the server is unreachable or the model is missing.
        """
        try:
            response = self.client.list()
        except Exception as exc:
            raise RuntimeError(f"Ollama is unreachable at {self.host}: {exc}") from exc
        models = [m.model for m in response.models]
        if not any(m == self.model or m.startswith(f"{self.model}:") for m in models):
            raise RuntimeError(
                f"Model {self.model!r} is not pulled on {self.host}. "
                f"Available: {models or 'none'}. "
                f"Pull it with: ollama pull {self.model}"
            )

    def caption_frame(self, image_path: Path) -> str:
        """Caption a single frame image.

        Args:
            image_path: Path to a PNG/JPEG frame.

        Returns:
            Caption text.
        """
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [str(image_path)],
                }
            ],
            keep_alive=self.keep_alive,
        )
        return response.message.content.strip()

    def caption_frames(self, frames: Sequence[FrameRef]) -> list[dict]:
        """Caption selected frames.

        Args:
            frames: Frame refs as ``(index, timestamp_s, path)``.

        Returns:
            List of ``{frame, timestamp_s, caption}`` dictionaries.
        """
        captions: list[dict] = []
        for idx, timestamp, path in frames:
            logger.info("Captioning frame %s (%s)", idx, path)
            start = time.monotonic()
            caption = self.caption_frame(path)
            captions.append({
                "frame": idx,
                "timestamp_s": timestamp,
                "caption": caption,
                "elapsed_s": round(time.monotonic() - start, 2),
            })
        return captions

    def unload(self) -> None:
        """Unload the model from VRAM (``keep_alive=0``)."""
        try:
            self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": ""}],
                keep_alive=0,
            )
        except Exception:
            logger.warning("Failed to unload Ollama model %s", self.model)


def caption_frames(
    frames: Sequence[FrameRef],
    flagged_frames: Sequence[dict],
    model: str = config.DEFAULT_OLLAMA_MODEL,
    host: str = config.DEFAULT_OLLAMA_HOST,
    top_k: int | None = None,
) -> list[dict]:
    """Convenience wrapper: select flagged frames and caption them.

    Args:
        frames: All extracted frames as ``(index, timestamp_s, path)``.
        flagged_frames: Aggregation output dicts with ``frame`` and ``score``.
        model: Ollama model name.
        host: Ollama server URL.
        top_k: Optional cap on frames captioned (highest scores first).

    Returns:
        List of ``{frame, timestamp_s, caption}`` dictionaries.
    """
    captioner = OllamaCaptioner(model=model, host=host)
    selected = select_flagged_frames(frames, flagged_frames, top_k=top_k)
    return captioner.caption_frames(selected)
