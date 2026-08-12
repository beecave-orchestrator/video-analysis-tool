"""Tests for the Ollama VLM backend (mocked) plus an optional integration test."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from video_nsfw_tagger import config
from video_nsfw_tagger import ollama as ollama_mod
from video_nsfw_tagger.ollama import OllamaCaptioner, select_flagged_frames

FRAMES = [
    (1, 0.0, Path("/tmp/f1.png")),
    (2, 1.0, Path("/tmp/f2.png")),
    (3, 2.0, Path("/tmp/f3.png")),
    (4, 3.0, Path("/tmp/f4.png")),
]

FLAGGED = [
    {"frame": 2, "timestamp_s": 1.0, "score": 0.9},
    {"frame": 4, "timestamp_s": 3.0, "score": 0.8},
    {"frame": 1, "timestamp_s": 0.0, "score": 0.7},
]


def _mock_client(model_names=(config.DEFAULT_OLLAMA_MODEL,), caption="a caption"):
    client = MagicMock()
    client.list.return_value = SimpleNamespace(
        models=[SimpleNamespace(model=m) for m in model_names]
    )
    client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=caption))
    return client


def test_select_flagged_frames_orders_by_score_desc():
    selected = select_flagged_frames(FRAMES, FLAGGED)
    assert [f[0] for f in selected] == [2, 4, 1]


def test_select_flagged_frames_top_k_keeps_highest_scores():
    selected = select_flagged_frames(FRAMES, FLAGGED, top_k=2)
    assert [f[0] for f in selected] == [2, 4]


def test_select_flagged_frames_maps_frame_to_path():
    selected = select_flagged_frames(FRAMES, FLAGGED, top_k=1)
    assert selected == [(2, 1.0, Path("/tmp/f2.png"))]


def test_select_flagged_frames_ignores_unknown_indices():
    flagged = [{"frame": 99, "timestamp_s": 98.0, "score": 0.99}]
    assert select_flagged_frames(FRAMES, flagged) == []


def test_select_flagged_frames_empty():
    assert select_flagged_frames(FRAMES, []) == []


def test_check_available_success():
    captioner = OllamaCaptioner()
    captioner.client = _mock_client((config.DEFAULT_OLLAMA_MODEL,))
    captioner.check_available()  # should not raise


def test_check_available_accepts_untagged_model_name():
    captioner = OllamaCaptioner(model="qwen3-vl")
    captioner.client = _mock_client(("qwen3-vl:latest",))
    captioner.check_available()  # should not raise


def test_check_available_fails_when_model_missing():
    captioner = OllamaCaptioner(model="qwen3-vl:4b")
    captioner.client = _mock_client(("llama3:8b",))
    with pytest.raises(RuntimeError, match="not pulled"):
        captioner.check_available()


def test_check_available_fails_when_server_unreachable():
    captioner = OllamaCaptioner()
    client = MagicMock()
    client.list.side_effect = ConnectionError("refused")
    captioner.client = client
    with pytest.raises(RuntimeError, match="unreachable"):
        captioner.check_available()


def test_caption_frame_sends_prompt_and_image():
    captioner = OllamaCaptioner(model="qwen3-vl:4b")
    captioner.client = _mock_client(caption="  some caption  ")
    result = captioner.caption_frame(Path("/tmp/f1.png"))
    assert result == "some caption"
    kwargs = captioner.client.chat.call_args.kwargs
    assert kwargs["model"] == "qwen3-vl:4b"
    assert kwargs["messages"][0]["images"] == ["/tmp/f1.png"]
    assert kwargs["messages"][0]["content"] == ollama_mod.DEFAULT_PROMPT
    assert kwargs["keep_alive"] == ollama_mod.KEEP_ALIVE


def test_caption_frames_structure():
    captioner = OllamaCaptioner()
    captioner.client = _mock_client()
    selected = select_flagged_frames(FRAMES, FLAGGED, top_k=2)
    captions = captioner.caption_frames(selected)
    assert [c["frame"] for c in captions] == [2, 4]
    assert [c["timestamp_s"] for c in captions] == [1.0, 3.0]
    assert [c["caption"] for c in captions] == ["a caption", "a caption"]
    assert all(c["elapsed_s"] >= 0 for c in captions)
    # Gating check: exactly the selected frames were captioned, no others.
    assert captioner.client.chat.call_count == 2


def test_module_caption_frames_end_to_end_mocked():
    with patch.object(OllamaCaptioner, "__init__", lambda self, **kw: None):
        with patch.object(
            OllamaCaptioner, "caption_frames", return_value=[{"frame": 2}]
        ) as mock_cf:
            result = ollama_mod.caption_frames(FRAMES, FLAGGED, top_k=1)
    assert result == [{"frame": 2}]
    (called_frames,), _ = mock_cf.call_args
    assert [f[0] for f in called_frames] == [2]


def test_unload_uses_keep_alive_zero():
    captioner = OllamaCaptioner()
    captioner.client = _mock_client()
    captioner.unload()
    assert captioner.client.chat.call_args.kwargs["keep_alive"] == 0


def test_unload_swallows_errors():
    captioner = OllamaCaptioner()
    client = MagicMock()
    client.chat.side_effect = ConnectionError("gone")
    captioner.client = client
    captioner.unload()  # should not raise


@pytest.mark.integration
def test_ollama_integration_caption_real_frame(tmp_path):
    """Caption a real synthetic frame; skipped when Ollama/model unavailable."""
    pytest.importorskip("ollama")
    try:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        frame = tmp_path / "frame.png"
        img.save(frame)

        captioner = OllamaCaptioner()
        captioner.check_available()
        caption = captioner.caption_frame(frame)
    except RuntimeError as exc:
        pytest.skip(f"Ollama unavailable: {exc}")
    assert isinstance(caption, str) and caption
