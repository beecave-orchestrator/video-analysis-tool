# Video NSFW Analysis Pipeline — Megaplan

Build a local, offline, privacy-first CLI tool that samples video frames with ffmpeg, scores them with the Falconsai NSFW ViT, gates NSFW-positive frames through a pluggable local VLM (default Qwen2.5-VL-3B) for captions, derives act tags via an offline keyword lexicon, and writes JSON sidecars plus a per-directory SQLite index — phased as ViT MVP first (Phase A), then the VLM stage (Phase B), targeting Linux + AMD RX 6600/ROCm with CPU fallback.

- **Source plan:** `2026-07-09-video-nsfw-analysis-pipeline.md` (workspace root)
- **Feasibility:** verified in prior session (see "Feasibility findings" below)
- **Development constraint:** Primary development and the PoC happen on an Intel MacBook (CPU only). **Heavy dependencies — especially the 3B VLM — are NOT installed on this machine.** The project moves to the AMD RX 6600 / ROCm server for VLM validation and Phase B.
- **Status:** ready for implementation after review

---

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | v1 feature set | Broad NSFW gate + **VLM captions** (act descriptions second stage) |
| 2 | Primary target | **Linux + RX 6600 (gfx1032) + ROCm**; Intel Mac for light CPU smoke tests only |
| 3 | Output | **JSON sidecar per video + SQLite index** |
| 4 | SQLite location | `./video_nsfw_index.db` in cwd / next to first input; `--db PATH` override |
| 5 | VLM invocation | **NSFW-gated only** — VLM runs solely on frames above the NSFW threshold (plus optional top-K cap) |
| 6 | VLM model | **Pluggable: default `Qwen/Qwen2.5-VL-3B-Instruct`**, override via `--vlm-model` |
| 7 | Act-tag derivation | **Keyword/phrase lexicon parse** of captions (deterministic, editable, no extra model) |
| 8 | Interface | **CLI only** for v1 (Gradio deferred) |
| 9 | Packaging | **PDM** (`pyproject.toml` + `pdm.lock`, console script entrypoint) |
| 10 | Phasing | **Phase A: ViT MVP → Phase B: VLM stage** (validate ROCm before loading the 3B VLM) |

## Objective

Ship `video-nsfw-tagger` as an installable CLI:

- **Phase A (MVP):** `vnt scan <path>` — ffmpeg frame sampling at 1 fps → Falconsai binary NSFW score per frame → aggregate (`nsfw_percent`, `max_score`, `flagged_frames[]`, `verdict` at configurable threshold) → JSON sidecar (`video.mp4` → `video.nsfw.json`) + upsert into `./video_nsfw_index.db`. Runs on CPU (Mac dev) and ROCm (target).
- **Phase B (VLM):** opt-in `--vlm` flag — for frames above threshold, caption with local VLM → parse captions against a lexicon file → merge `captions[]` and `act_tags[]` into sidecar + DB.

## Feasibility findings (verified, prior session)

| Component | Result | Evidence |
|-----------|--------|----------|
| ffmpeg 8.1.2 frame extraction @ 1 fps | **Pass** | 3s clip → 3 PNGs in ~20 ms |
| `Falconsai/nsfw_image_detection` (ViT, Apache-2.0, ~327 MB) | **Pass** | CPU pipeline; synthetic frame scored `normal` 0.9996; ~0.5 s first frame, ~0.21 s/frame batched |
| CLIP zero-shot (`open_clip` ViT-B-32) | **Weak** | High NSFW mass on SFW test pattern → unreliable alone; not in v1 path |
| `Qwen/Qwen2.5-VL-3B-Instruct` | **Not run; do NOT install on this Mac** | Sharded safetensors on HF; too heavy for an i5-8257U/Iris 645. **Install and validate only on the RX 6600 / ROCm server after the PoC moves.** |
| Offline after first download | **Yes** | Models cache under `~/.cache/huggingface` |

**Throughput estimate (Intel Mac CPU):** 1-min video ≈ 12–30 s; 1-hour video ≈ 12–30 min. Expect ~5–20× speedup on RX 6600/ROCm (must be confirmed on target).

## Environment mismatch & stack constraints

| Plan assumed | Dev machine (this Mac) |
|--------------|------------------------|
| Linux + RX 6600 (gfx1032) + ROCm | macOS 15.7.7, i5-8257U, Iris Plus 645, **CPU only** |
| GPU PyTorch | x86_64 macOS torch tops out at **2.2.2**, no MPS |

Pins discovered during feasibility:

- **Python 3.12** (system python3 is 3.14 — torch incompatible)
- `torch==2.2.2` (CPU wheel) on the Mac; official ROCm torch wheels on the Linux target
- `transformers>=4.40,<5` (transformers 5.x requires torch ≥2.4)
- `numpy<2` (torch 2.2.2 compiled against NumPy 1.x)
- `pillow`, `safetensors`; ffmpeg/ffprobe on PATH
- **CLI / terminal UX:** `typer[all]>=0.16.0` (provides `typer` + `rich` for shell completion, rich help, and progress output) and `rich` explicitly declared for custom console/progress UI
- **Mac dev constraint:** the `pyproject.toml`/pdm lock for Phase A must include **only the Falconsai ViT stack**. Do not include `qwen-vl-utils`, `accelerate` for VLM, or any other VLM-only dependencies until the project is moved to the ROCm server.

## Architecture

```
src/video_nsfw_tagger/
├── __init__.py
├── cli.py            # Typer CLI: scan, report, config-show; uses Annotated[...] options, version callback, lazy imports (per Typer CLI standard)
├── config.py         # defaults + CLI overrides (threshold, fps, db path, models)
├── device.py         # device resolve: cuda(ROCm) → mps → cpu; HSA_OVERRIDE_GFX_VERSION doc
├── extract.py        # ffmpeg/ffprobe subprocess: duration probe, 1 fps PNG frames to temp dir, cleanup
├── classify.py       # Falconsai ViT wrapper (transformers pipeline, batched)
├── aggregate.py      # per-video stats: nsfw_percent, max_score, verdict, flagged_frames
├── sidecar.py        # read/write video.nsfw.json (atomic write)
├── db.py             # sqlite3 stdlib: schema init, upsert, query helpers
├── vlm.py            # Phase B: pluggable VLM captioner (Qwen2.5-VL default)
└── lexicon.py        # Phase B: load lexicon YAML/JSON, caption → act_tags
lexicon/
└── acts.yaml         # editable keyword/phrase → tag mapping
tests/
├── test_aggregate.py
├── test_sidecar.py
├── test_db.py
├── test_lexicon.py   # Phase B
└── fixtures/         # synthetic ffmpeg-generated clips (no real content)
pyproject.toml        # PDM; [project.scripts] vnt = "video_nsfw_tagger.cli:app"
pdm.lock
README.md
```

## Starting `pyproject.toml` template

The following PDM/Typer/Ruff boilerplate is the starting point for the project. During Phase A, the `name`, `requires-python`, and `dependencies` fields are customized for `video-nsfw-tagger` and the Falconsai stack; the lint/test tooling stays unchanged.

```toml
[project]
name = "python-cli-boilerplate"
version = "0.1.0"
description = "Reusable boilerplate for Python CLI applications"
readme = "README.md"
requires-python = "~=3.10"
license = { file = "LICENSE" }
dependencies = [
    "typer==0.27.0",
    "rich==15.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0.2",
    "pytest-cov>=7.0.0",
    "ruff>=0.14.11",
    "typing-extensions>=4.15.0",
    "docstr-coverage>=2.3.2",
    "interrogate>=1.7.0",
]

[tool.pdm]
distribution = false

[tool.pdm.scripts]
lint = "ruff check ."
format = "ruff format ."
format-check = "ruff format --check ."
fix = "ruff check --fix ."
test = "pytest -q"
test-cov = "pytest --cov=. --cov-report=term-missing:skip-covered --cov-report=xml"

[tool.ruff]
line-length = 88
target-version = "py310"
preview = true

[tool.ruff.lint]
select = [
    "F",
    "E",
    "W",
    "N",
    "I",
    "D",
    "DOC",
    "ANN",
    "TID",
    "UP",
    "FA",
]
extend-select = ["ANN401"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "integration: mark tests that hit real external integrations or slower paths",
]
```

## JSON sidecar schema (v1)

```json
{
  "schema_version": 1,
  "video_path": "sample.mp4",
  "duration_s": 60.0,
  "fps_sampled": 1,
  "frames_total": 60,
  "threshold": 0.7,
  "device": "cpu",
  "model": "Falconsai/nsfw_image_detection",
  "nsfw_percent": 12.3,
  "max_score": 0.94,
  "verdict": "nsfw",
  "flagged_frames": [{"frame": 7, "timestamp_s": 7.0, "score": 0.94}],
  "vlm": {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "captions": [{"frame": 7, "caption": "..."}],
    "act_tags": ["tag1", "tag2"]
  },
  "created_at": "2026-08-09T19:00:00+02:00"
}
```

`vlm` block is `null`/absent in Phase A output.

## SQLite schema (`video_nsfw_index.db`)

```sql
CREATE TABLE IF NOT EXISTS videos (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  duration_s REAL,
  frames_total INTEGER,
  nsfw_percent REAL,
  max_score REAL,
  verdict TEXT,
  threshold REAL,
  model TEXT,
  vlm_model TEXT,
  act_tags TEXT,          -- JSON array
  sidecar_path TEXT,
  scanned_at TEXT
);
```

## CLI spec

```bash
vnt scan <file|dir> [--recursive] [--threshold 0.7] [--fps 1] [--max-duration N]
         [--db PATH] [--device auto|cpu|cuda|mps]
         [--vlm] [--vlm-model HF_ID] [--vlm-top-k N] [--lexicon PATH]
vnt report [--db PATH] [--verdict nsfw] [--min-percent 10]
```

- `scan` on a directory batches all supported video files; writes sidecar next to each video and upserts into the DB (default `./video_nsfw_index.db`). Rich console for progress output.
- `--vlm` is a no-op with a printed notice in Phase A builds (flag scaffolded in Phase B).

## Implementation plan

### Phase A — ViT MVP (develop and verify on this Mac; no VLM deps)

- [ ] `pdm init` project: Python 3.12 requires-python, deps pinned for **Mac Phase A** (`typer[all]`, `rich`, `torch==2.2.2`, `transformers>=4.40,<5`, `numpy<2`, `pillow`, `safetensors` — no VLM/Qwen/accelerate-VLM, no `torch>=2.4`); console script `vnt` and `python -m video_nsfw_tagger` entry points
- [ ] `device.py` — resolve `auto`: cuda→mps→cpu; expose torch/ROCm info for `vnt config-show`
- [ ] `extract.py` — ffprobe duration; ffmpeg `-vf fps=1` into `tempfile.TemporaryDirectory`; yield (index, timestamp, path); guarantee cleanup
- [ ] `classify.py` — lazy-load pipeline once; batch inference (batch size flag, default 8)
- [ ] `aggregate.py` + `sidecar.py` — stats, verdict, atomic sidecar write
- [ ] `db.py` — stdlib sqlite3, idempotent schema, upsert by `path`
- [ ] `cli.py` — `scan` (file/dir/recursive), `report`, `config-show`
- [ ] Tests: pure-logic units (aggregate, sidecar round-trip, db upsert) + one synthetic ffmpeg clip integration test; no network, no real content
- [ ] README: install (PDM), quickstart, threshold tuning notes, privacy statement

**Phase A acceptance criteria:**

- `vnt scan` on a synthetic clip produces a valid sidecar matching the schema and a DB row; exit 0 on CPU-only Mac.
- Re-scanning the same file updates (not duplicates) the DB row.
- Temp frames are removed after every run (success and failure paths).
- Unit + integration tests pass offline (`HF_HUB_OFFLINE=1` after first model download).

### Phase B — VLM stage (develop against ROCm target; CPU optional/slow)

- [ ] Validate ROCm on Linux target: official ROCm torch wheel, `torch.cuda.is_available()`, `HSA_OVERRIDE_GFX_VERSION=gfx1030` if needed, `rocm-smi` during load
- [ ] `vlm.py` — pluggable captioner protocol; default `Qwen/Qwen2.5-VL-3B-Instruct`; `--vlm-model` override; NSFW-gated invocation only (frames ≥ threshold, optional `--vlm-top-k` cap by score). **First test on the AMD RX 6600 / ROCm server; on this Mac the VLM code is structurally implemented but the tests skip unless `--device cuda` is available or explicit CI server flag is set.**
- [ ] `lexicon.py` + `lexicon/acts.yaml` — keyword/phrase → tag mapping; case-insensitive phrase match; editable without code changes
- [ ] Merge `captions[]` / `act_tags[]` into sidecar `vlm` block and DB columns
- [ ] Throughput benchmark on RX 6600 (ViT path and gated VLM path); document results in README

**Phase B acceptance criteria:**

- `vnt scan --vlm` on a flagged clip adds non-empty `vlm.captions` to the sidecar and lexicon-derived `act_tags` where matched.
- VLM never runs on frames below the NSFW threshold (verified by log/call count in tests).
- ROCm run shows GPU utilization via `rocm-smi`; CPU fallback path still works with `--device cpu`.

## Out of scope

- Fine-tuning on private datasets; custom act classifiers
- Cloud APIs or any data upload (hard requirement: fully offline)
- CLIP zero-shot detector (tested weak — deferred unless gated+calibrated later)
- FlowNSFW / optical flow (maybe later; not needed for MVP)
- Gradio UI, filename renaming/prefixing
- Production deployment / legal-compliance review (user responsibility)
- Minor-related content handling — hard decline rule stays outside the pipeline

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ROCm/PyTorch on gfx1032 | Medium | High | `HSA_OVERRIDE_GFX_VERSION=gfx1030`; official ROCm wheels; CPU fallback documented; validate in Phase B step 1 |
| VLM too heavy for 8 GB VRAM | Medium | Medium | 3B default; quantize/offload if OOM; `--vlm-model` swap to lighter model |
| Act-tag false positives from lexicon | High | Low–Med | Deterministic lexicon is auditable/editable; captions stored alongside tags for user review |
| Long-video runtime | Medium | Medium | 1 fps sampling, batch inference, `--max-duration`, GPU path |
| Model bias / accuracy unknown on real corpora | Medium | Medium | Only smoke-tested; user tunes `--threshold` on own samples; document limitation |

## Verification commands

```bash
# Setup (dev Mac)
pdm install
pdm run vnt config-show          # torch version, device resolution

# Synthetic end-to-end (offline-safe)
ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=30 -pix_fmt yuv420p -y /tmp/vnt-sample.mp4
pdm run vnt scan /tmp/vnt-sample.mp4
cat /tmp/vnt-sample.nsfw.json
pdm run vnt report

# Tests
pdm run pytest -q

# ROCm target (Phase B)
rocm-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
pdm run vnt scan --vlm /path/to/sample.mp4
```

## Notes

- All processing local; the "decline if minor-related" rule is a hard boundary outside the pipeline.
- Test fixtures must be synthetic (ffmpeg-generated patterns) — no real NSFW/SFW content in the repo or tests.
- HF models are public and unauthenticated; first run downloads ~327 MB (ViT) / ~6 GB (VLM) into `~/.cache/huggingface`.
- Follow the repo's writing-style guide for README/docs (warm, clear, direct, practical).
