# video-nsfw-tagger — Project overview

## Summary

Local, offline, privacy-first CLI that samples video frames, scores each frame
with the Falconsai NSFW ViT, and writes a JSON sidecar plus an SQLite index.

- **Phase A (current):** ViT MVP — `vnt scan`, `vnt report`, `vnt config-show`.
- **Phase B (future):** gated local VLM captions and act-tag lexicon.

## Technology stack

- Python 3.12
- PDM-packaged `pyproject.toml` / `setuptools` editable install
- Typer + Rich for the CLI
- PyTorch 2.2.2 / Transformers 4.57 / Pillow / NumPy 1.26
- ffmpeg / ffprobe for frame extraction
- sqlite3 stdlib for the index

## Repository layout

```text
video-analysis-tool/
├── pyproject.toml
├── README.md
├── LICENSE
├── project-overview.md      <- this file
├── lexicon/acts.yaml        <- Phase B keyword lexicon
├── src/video_nsfw_tagger/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py               # Typer app (scan, report, config-show)
│   ├── config.py            # defaults and constants
│   ├── device.py            # device auto-resolution (cuda/mps/cpu)
│   ├── extract.py           # ffprobe + ffmpeg frame extraction
│   ├── classify.py          # Falconsai ViT pipeline wrapper
│   ├── aggregate.py         # per-video stats and verdict
│   ├── sidecar.py           # atomic JSON sidecar I/O
│   ├── db.py                # SQLite schema / upsert / query
│   ├── vlm.py               # Phase B VLM placeholder
│   └── lexicon.py           # Phase B lexicon helpers
└── tests/
    ├── conftest.py
    ├── test_aggregate.py
    ├── test_sidecar.py
    ├── test_db.py
    ├── test_lexicon.py
    └── test_extract.py
```

## Entry points

- `vnt` console script
- `python -m video_nsfw_tagger`
- `pdm run vnt` (PDM scripts not yet used, but console script is installed)

## Verification commands

```bash
# installed in the local .venv
pip install -e .
vnt config-show
vnt scan /tmp/vnt-sample.mp4
vnt report

# tests
pytest -q
pytest -q tests/test_extract.py   # ffmpeg integration / synthetic video
```

## Current status

- Phase A modules implemented and installed.
- Unit tests (aggregate, sidecar, db, lexicon) pass.
- ffmpeg extraction integration test passes.
- End-to-end `vnt scan` on a synthetic 5 s clip produces a valid sidecar and a
  SQLite row using the cached Falconsai ViT.
- `--vlm` is a no-op in Phase A; the 3B VLM remains for ROCm server validation.
