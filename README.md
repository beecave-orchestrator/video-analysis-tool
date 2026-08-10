# video-nsfw-tagger

A local, offline, privacy-first CLI for NSFW video analysis.

`vnt scan <path>` samples frames with `ffmpeg`, scores each frame with the
[Falconsai NSFW ViT](https://huggingface.co/Falconsai/nsfw_image_detection),
and writes a JSON sidecar plus a per-directory SQLite index.

Phase A ships the ViT pipeline. Phase B (VLM captioning via Ollama) is
validated on the AMD RX 6600 / ROCm server — see
[docs/rocm-validation.md](docs/rocm-validation.md) for environment details.

## Quick start (ROCm target)

```bash
# Create venv with ROCm torch
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install -e ".[dev]"

# gfx1032 (RX 6600) workaround — required before every run
export HSA_OVERRIDE_GFX_VERSION=10.3.0

vnt config-show

# Synthetic end-to-end smoke test
ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=30 -pix_fmt yuv420p -y /tmp/vnt-sample.mp4
vnt scan /tmp/vnt-sample.mp4
cat /tmp/vnt-sample.nsfw.json
vnt report
```

## Usage

```bash
vnt scan <file|dir> [--recursive] [--threshold 0.7] [--fps 1]
         [--max-duration N] [--db PATH] [--device auto|cpu|cuda|mps]
         [--vlm] [--vlm-model HF_ID] [--vlm-top-k N] [--lexicon PATH]
vnt report [--db PATH] [--verdict nsfw] [--min-percent 10]
vnt config-show
```

Use `--vlm` only on the ROCm target; it is a no-op in Phase A builds.

## Development

```bash
source .venv/bin/activate
pytest -q          # unit tests (no model downloads)
ruff check .       # lint
ruff format .      # format
```

## Privacy note

All processing is local. Models are downloaded from Hugging Face on first use and
cached under `~/.cache/huggingface`; after the first run the tool can operate
offline with `HF_HUB_OFFLINE=1`.
