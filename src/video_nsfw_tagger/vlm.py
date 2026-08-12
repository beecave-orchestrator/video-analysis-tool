"""Pluggable local VLM captioner (Phase B placeholder)."""

from pathlib import Path

DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def caption_frames(
    image_paths: list[Path],
    model_name: str = DEFAULT_VLM_MODEL,
    device: str = "cpu",
    top_k: int | None = None,
) -> list[dict]:
    """Caption the provided frames using a local VLM.

    This is a Phase B placeholder. It is intentionally not wired in Phase A
    because the default 3B VLM is too heavy for the development Mac and is
    intended for the RX 6600 / ROCm server.

    Args:
        image_paths: Frames to caption.
        model_name: Hugging Face VLM ID.
        device: Torch device identifier.
        top_k: Optional maximum number of flagged frames to caption.

    Returns:
        List of ``{frame, timestamp_s, caption}`` dictionaries.

    Raises:
        NotImplementedError: In Phase A builds.
    """
    raise NotImplementedError(
        "VLM captioning is a Phase B feature and is not available in this build."
    )
