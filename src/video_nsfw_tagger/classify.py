"""Falconsai NSFW ViT wrapper with batched inference."""

import logging
from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from transformers import Pipeline, pipeline

from video_nsfw_tagger import config

logger = logging.getLogger(__name__)

DEFAULT_VIT_MODEL = config.DEFAULT_VIT_MODEL


def load_pipeline(
    device: str | int,
    model_name: str = DEFAULT_VIT_MODEL,
    batch_size: int = 8,
) -> Pipeline:
    """Load the image-classification pipeline for the given model.

    Args:
        device: Torch device identifier.
        model_name: Hugging Face model ID.
        batch_size: Batch size forwarded to the pipeline ``__call__``.

    Returns:
        A ``transformers`` image-classification pipeline.
    """
    token = config.load_hf_token()
    if token is None:
        logger.warning(
            "No HF token found for %s; gated models will fail to load. "
            "Set HF_TOKEN or add it to %s.",
            model_name,
            config.HF_TOKEN_ENV_PATH,
        )
    logger.info("Loading NSFW ViT %s on %s", model_name, device)
    return pipeline(
        "image-classification",
        model=model_name,
        device=device,
        batch_size=batch_size,
        token=token,
    )


def _nsfw_score(result: list[dict]) -> float:
    """Extract the ``nsfw`` label score from a pipeline result entry.

    Args:
        result: Label/score entries for one image.

    Returns:
        The NSFW score, or 0.0 if no ``nsfw`` label is present.
    """
    for entry in result:
        if entry.get("label", "").lower() == "nsfw":
            return float(entry["score"])
    return 0.0


def classify_batch(
    pipe: Pipeline,
    image_paths: Sequence[Path],
    batch_size: int = 8,
) -> list[float]:
    """Score a list of frame paths, returning one NSFW score per frame.

    Args:
        pipe: Loaded image-classification pipeline.
        image_paths: Paths to frame PNG/JPEG files.
        batch_size: Inference batch size.

    Returns:
        NSFW scores in the same order as ``image_paths``.
    """
    if not image_paths:
        return []

    images = [Image.open(p).convert("RGB") for p in image_paths]
    results = pipe(images, batch_size=batch_size, top_k=2)

    # The pipeline returns a single list of dicts for one image and a list of
    # lists for multiple images; normalise to a list of lists.
    if isinstance(results[0], dict):
        results = [results]

    return [_nsfw_score(r) for r in results]
