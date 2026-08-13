# AGENTS.md

## Purpose

`video-nsfw-tagger` is a local, offline, privacy-first Python CLI for analysing
video files. It samples frames with `ffmpeg`, runs a fast NSFW Vision
Transformer (ViT) against every sampled frame, optionally captions only
flagged frames through a locally served Ollama vision-language model (VLM),
and writes a JSON sidecar plus a SQLite index.

Keep processing local: frames, captions, and database records are user content.
Do not add network uploads, telemetry, or logging that exposes video paths,
frame contents, captions, Hugging Face tokens, or other secrets.

## Setup and commands

### Prerequisites

- Python `>=3.12,<3.14`
- `ffmpeg` and `ffprobe` on `PATH`
- For AMD GPU acceleration: ROCm-compatible PyTorch and an AMD GPU
- For `--vlm`: a reachable local Ollama server with the selected vision model
  already pulled

The primary target is an AMD Radeon RX 6600 (`gfx1032`, 8 GB VRAM). On that
host, this workaround must be exported before Python or `vnt` starts:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Install

PDM is the repository's dependency manager and lockfile owner. Its scoped
`pytorch-rocm` source ensures PyTorch packages resolve from the ROCm index,
not CUDA-only PyPI wheels:

```bash
pdm sync -G dev
```

For a fresh virtual environment, install ROCm PyTorch first, then the editable
project with development dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install -e ".[dev]"
```

Do not replace the scoped ROCm package source or allow `torch` / `torchvision`
to resolve from normal PyPI on the AMD target. Verify the environment with:

```bash
pdm run vnt config-show
# or, in an activated editable environment:
vnt config-show
```

The default ViT model, `Falconsai/nsfw_image_detection_26`, is gated on
Hugging Face. It obtains a read token from `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, or
`~/.config/insanely-fast-whisper-rocm/.env`. Never print or commit that token.
The Hugging Face model terms must be accepted before its first download.

### Tests

```bash
pdm run test
pdm run pytest -q tests/test_ollama.py
pdm run pytest -q -m "not integration and not slow"
pdm run test-cov
```

The default suite includes marker-labelled cases: the extraction tests invoke
real `ffmpeg`, and the Ollama integration test attempts a real local caption
when its server and default model are available (otherwise it skips). Use the
marker-excluding command above when a fully local, service-free fast check is
needed; do not assume markers are excluded automatically.

### Lint and format

```bash
pdm run lint
pdm run format-check
pdm run format       # applies formatting
pdm run fix          # applies Ruff lint fixes
```

Ruff uses an 88-character line length, Google-style docstrings, and strict
checks for production code. Tests intentionally ignore the docstring and
annotation rule sets; preserve the tighter standard in `src/`.

### CLI smoke test

```bash
ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=30 \
  -pix_fmt yuv420p -y /tmp/vnt-sample.mp4
pdm run vnt scan /tmp/vnt-sample.mp4
pdm run vnt report
```

The first non-cached scan downloads or loads the gated ViT model. Do not use a
real user video for a basic code-path smoke test.

## Project structure

```text
.
├── pyproject.toml                    # project metadata, PDM sources/scripts, Ruff, pytest
├── pdm.lock                          # locked dependency graph; keep in sync with pyproject.toml
├── README.md                         # user-facing setup and CLI usage
├── docs/
│   ├── rocm-validation.md            # measured AMD/ROCm environment notes
│   └── prompt-experiments.md         # VLM prompt-sweep workflow
├── lexicon/acts.yaml                 # tag -> phrase vocabulary for VLM captions
├── experiments/
│   ├── prompts/                      # candidate Markdown prompt definitions
│   ├── cache/                        # generated frame/score caches; do not edit as source
│   └── results/                      # generated prompt-sweep artefacts; do not edit as source
├── scripts/10-prompt-test.sh         # repeatable multi-prompt VLM experiment driver
├── src/video_nsfw_tagger/
│   ├── cli.py                        # Typer commands and orchestration
│   ├── config.py                     # defaults, supported extensions, safe HF-token lookup
│   ├── device.py                     # auto/cpu/cuda/mps device resolution
│   ├── extract.py                    # ffprobe duration + temporary ffmpeg frame extraction
│   ├── classify.py                   # gated Falconsai ViT loading and batched scoring
│   ├── aggregate.py                  # score statistics, flagged frames, and verdict
│   ├── framecache.py                 # durable frame/score cache for prompt experiments
│   ├── ollama.py                     # concrete local Ollama VLM backend
│   ├── vlm.py                        # reserved placeholder for a future direct-transformers backend
│   ├── lexicon.py                    # YAML/JSON vocabulary loading and phrase matching
│   ├── sidecar.py                    # atomic <video>.nsfw.json read/write helpers
│   ├── db.py                         # SQLite schema, upserts, reporting queries
│   └── prompt_report.py              # prompt-sweep aggregation and report generation
└── tests/                            # pytest unit tests plus marker-labelled integration/slow cases
```

The installed console entry point is `vnt = video_nsfw_tagger.cli:app`; the
module entry point is `python -m video_nsfw_tagger`.

## Architecture and data flow

### Standard scan

`vnt scan` follows this ordered pipeline:

1. `cli._discover_videos()` finds only extensions from `config.SUPPORTED_EXTS`.
2. `extract.extracted_frames()` creates a temporary directory, invokes
   `ffmpeg`, and yields `(frame_index, timestamp_s, image_path)` tuples.
3. `classify.load_pipeline()` loads the Falconsai image classifier once per
   cache-miss workflow; `classify.classify_batch()` produces one NSFW score per
   frame.
4. `aggregate.aggregate()` determines video statistics, verdict, and the
   score-bearing `flagged_frames` list using the configured threshold.
5. With `--vlm`, `ollama.select_flagged_frames()` orders flagged frames by
   descending score, optionally limits them with `--vlm-top-k`, then
   `OllamaCaptioner` captions them. `lexicon.find_matches()` derives act tags.
6. `sidecar.write_sidecar()` writes `<video-stem>.nsfw.json` atomically and
   `db.upsert_video()` updates the selected SQLite index by video path.

Frame paths from `extracted_frames()` disappear at the end of its context.
Perform captioning while that context is still open. Cached scans are the
exception: their frame paths point to the persistent cache directory.

### Outputs and re-scan behaviour

- The sidecar contains scan metadata, scores, verdict, flagged frames, and a
  nullable `vlm` block.
- The VLM block records backend/model/host/prompt metadata, captions, matched
  phrases, and aggregate act tags.
- The SQLite `videos` table is an upsert index keyed by the video-path string
  supplied to the scan; it stores summary fields and serialized act tags, not
  all frame detail. Keep path representation consistent for reliable re-scans.
- A non-VLM re-scan preserves a readable existing sidecar's `vlm` block.
  Maintain this behaviour when editing persistence code.

### VLM, cache, and VRAM behaviour

The concrete VLM backend is `ollama.py`; retain `vlm.py` as the separate
placeholder for a future direct-Transformers implementation. The VLM is an
optional, expensive second stage—never caption every extracted frame by
default.

`OllamaCaptioner.check_available()` must fail fast before scan work if Ollama
is unreachable or the requested model is not present. The default request
timeout is 300 seconds, hidden reasoning is off, one retry is allowed per
failed caption, and each request keeps the model warm for 10 minutes.
`--keep-vlm-loaded` suppresses the scan's final unload. Individual caption
failures degrade gracefully: retain successful captions and persist the scan
rather than discarding the full video batch.

`--frame-cache` persists frames and ViT scores keyed by canonical video path,
size, mtime, FPS, maximum duration, and ViT model. The NSFW threshold is
intentionally excluded, so it can vary across prompt experiments without a
cache miss. Do not change cache-key inputs casually; doing so affects
reproducibility and cache correctness.

On the 8 GB RX 6600, use `--unload-vit-before-vlm` when GPU memory may be
tight. The prompt experiment driver also warns when other GPU processes are
already consuming significant VRAM and keeps the VLM loaded between prompts
except after the final prompt.

### Prompt experiments

The supported prompt-sweep workflow is:

```bash
scripts/10-prompt-test.sh watch/sample.mp4 --top-k 2
pdm run vnt prompt-report experiments/results/<timestamp>
```

The script copies the exact prompt set into the timestamped result directory,
reuses `experiments/cache/`, stores per-prompt sidecars, and generates
`report.json` plus `report.md`. Begin with a small `--top-k`; a full sweep is
many real VLM calls. Treat tag counts as a ranking aid, then inspect caption
dumps for incorrect descriptions or keyword false positives.

## Code conventions

- Use Python 3.12-compatible type hints and standard `pathlib.Path` paths.
- Keep public functions documented with concise Google-style docstrings,
  including `Args`, `Returns`, and `Raises` when relevant.
- Prefer small, pure helpers in the domain modules and keep user-facing
  rendering/progress handling in `cli.py`.
- Use `typer.BadParameter` for invalid CLI option combinations and display
  operational failures through the Rich console before exiting non-zero.
- Pass paths and model configuration explicitly; rely on `config.py` for
  project defaults instead of duplicating strings in callers.
- Use `yaml.safe_load`; do not introduce unsafe deserialization.
- Lexicon matching is case-insensitive and uses whole-phrase boundaries.
  Add focused tests whenever changing matching rules or `acts.yaml`.
- Keep sidecar writes atomic and maintain backwards-compatible nullable
  fields where practical. Generated results must remain readable by
  `prompt_report`.
- Do not log secrets. `config.load_hf_token()` deliberately returns a token
  without exposing it in messages or exceptions.

## Testing guidance

Place tests in `tests/test_<module>.py`. Prefer unit tests that mock external
services and use `tmp_path` for filesystem fixtures. The shared
`synthetic_video` fixture invokes `ffmpeg`; mark tests requiring it as both
`integration` and `slow`, matching `tests/test_extract.py`.

For Ollama changes:

1. Test selection/order, availability errors, client request shape, retries,
   and unload behaviour with a mocked `ollama.Client`.
2. Add a narrowly scoped `@pytest.mark.integration` test only when real
   Ollama behaviour cannot be mocked meaningfully.
3. Verify that a failed caption does not discard already generated captions.
4. Do not make normal `pytest -q` require a model download, a running server,
   or user content.

For new CLI orchestration, test reusable helpers or injected/module-mocked
boundaries rather than loading the ViT or VLM in unit tests.

## Boundaries

### Always

- Run the relevant pytest subset and `pdm run lint` after code changes; run
  `pdm run format-check` for formatting-sensitive changes.
- Preserve the two-stage cost model: ViT scores all frames, VLM receives only
  selected flagged frames.
- Check the existing sidecar and database schemas before changing scan output.
- Update README or `docs/prompt-experiments.md` when user-visible CLI options,
  model prerequisites, or experiment output layout changes.
- Keep `pyproject.toml`, `pdm.lock`, and the installed dependency intent
  coherent when changing dependencies.

### Ask first

- Replacing either default model, changing NSFW thresholds/verdict semantics,
  or changing the tag lexicon in a way that can alter user classifications.
- Database schema migrations or non-backwards-compatible sidecar changes.
- Changing persistent cache-key semantics, generated-result layout, or
  experiment ranking metrics.
- Pulling multi-gigabyte models, running broad scans on real videos, or
  starting/stopping shared GPU services or Docker containers.

### Never

- Commit API/Hugging Face tokens, real video files, frame caches, sidecars,
  SQLite databases, or experiment results containing user content.
- Replace the ROCm wheel source with CUDA/PyPI wheels to make installation
  appear to work.
- Bypass Ollama availability checks or silently fall back to a remote model.
- Delete or overwrite user videos, source sidecars, database rows, caches, or
  experiment directories without explicit user confirmation.
- Convert the VLM stage into all-frame inference by default.

## Troubleshooting

| Symptom | Likely cause and action |
| --- | --- |
| `vnt` cannot load the ViT | Accept the model's Hugging Face gate terms and set `HF_TOKEN` or configure the documented `.env` source. Do not expose the token while diagnosing. |
| `torch.cuda.is_available()` is false on the RX 6600 | Export `HSA_OVERRIDE_GFX_VERSION=10.3.0` before starting Python and confirm that the installed torch build is `+rocm`, not a PyPI CUDA build. |
| `--vlm` fails before frame extraction | Ollama is unavailable or the requested model is not pulled. Check the local service and `ollama list`; use `ollama pull <model>` deliberately. |
| VLM captioning is unexpectedly slow or times out | The first request includes model load; reduce `--vlm-top-k`, retain the default retry policy, unload the ViT first if needed, and free competing GPU workloads. |
| Repeated prompt runs re-extract frames | Verify that FPS, max duration, ViT model, video size, and mtime are unchanged and reuse the same `--frame-cache` directory. |
| Unexpected act tags | Inspect caption text and its matched phrases in the sidecar/report. The lexicon is keyword-based, so wording and negation can affect outcomes; add a regression test before changing matcher semantics. |
| `pdm sync` selects an incorrect Torch build | Verify the scoped `[[tool.pdm.source]]` block in `pyproject.toml`, regenerate the lock only as part of an intentional dependency change, and do not work around it by weakening the source rules. |

## Git workflow

The repository uses conventional-style history with an optional scope and
emoji (for example, `feat ✨: ...`, `docs 📝: ...`, `chore 📦: ...`). Keep
commits focused and do not include generated or sensitive artefacts. Before
committing, inspect `git status`, review the full diff, and run the relevant
verification commands. Do not push, create PRs, alter git configuration, or
rewrite history unless explicitly requested.
