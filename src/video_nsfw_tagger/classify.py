"""Falconsai NSFW ViT wrapper with batched inference."""

import logging
from pathlib import Path
from typing import List, Sequence, Union

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_VIT_MODEL = "Falconsai/nsfw_image_detection"


def load_pipeline(
    device: Union[str, int],
    model_name: str = DEFAULT_VIT_MODEL,
    batch_size: int = 8,
):
    """Load the image-classification pipeline for the given model.

    Args:
        device: Torch device identifier.
        model_name: Hugging Face model ID.
        batch_size: Batch size forwarded to the pipeline ``__call__``.

    Returns:
        A ``transformers`` image-classification pipeline.
    """
    logger.info("Loading NSFW ViT %s on %s", model_name, device)
    return pipeline(
        "image-classification",
        model=model_name,
        device=device,
        batch_size=batch_size,
    )


def _nsfw_score(result: List[dict]) -> float:
    """Extract the ``nsfw`` label score from a pipeline result entry."""
    for entry in result:
        if entry.get("label", "").lower() == "nsfw":
            return float(entry["score"])
    return 0.0


def classify_batch(
    pipe,
    image_paths: Sequence[Path],
    batch_size: int = 8,
) -> List[float]:
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
