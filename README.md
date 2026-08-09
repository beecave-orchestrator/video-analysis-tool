# video-nsfw-tagger

A local, offline, privacy-first CLI for NSFW video analysis.

`vnt scan <path>` samples frames with `ffmpeg`, scores each frame with the
[Falconsai NSFW ViT](https://huggingface.co/Falconsai/nsfw_image_detection),
and writes a JSON sidecar plus a per-directory SQLite index.

Phase A (this version) ships the ViT pipeline. VLM captioning and offline act
lexicon support are scaffolded for Phase B on the AMD RX 6600 / ROCm server.

## Quick start

```bash
pdm install
pdm run vnt config-show

# Synthetic end-to-end smoke test
ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=30 -pix_fmt yuv420p -y /tmp/vnt-sample.mp4
pdm run vnt scan /tmp/vnt-sample.mp4
cat /tmp/vnt-sample.nsfw.json
pdm run vnt report
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
pdm run test       # unit tests (no model downloads)
pdm run lint
pdm run format
```

## Privacy note

All processing is local. Models are downloaded from Hugging Face on first use and
cached under `~/.cache/huggingface`; after the first run the tool can operate
offline with `HF_HUB_OFFLINE=1`.
