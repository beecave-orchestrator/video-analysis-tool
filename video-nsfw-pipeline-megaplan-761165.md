# Video NSFW Analysis Pipeline — Megaplan

Build a local, offline, privacy-first CLI tool that samples video frames with ffmpeg, scores them with the Falconsai NSFW ViT, gates NSFW-positive frames through a pluggable local VLM (default Qwen3-VL-4B via Ollama) for captions, derives act tags via an offline keyword lexicon, and writes JSON sidecars plus a per-directory SQLite index — phased as ViT MVP first (Phase A), then the VLM stage (Phase B), targeting Linux + AMD RX 6600/ROCm with CPU fallback.

- **Source plan:** `2026-07-09-video-nsfw-analysis-pipeline.md` (workspace root)
- **Feasibility:** verified in prior session (see "Feasibility findings" below)
- **Development constraint:** Phase A was developed on an Intel MacBook (CPU only). Phase B environment setup and validation completed on the AMD RX 6600 / ROCm server. The project is now **ROCm-only** — the Mac dev constraint no longer applies.
- **Status:** Phase A complete, Phase B code implemented and verified end-to-end on real video (2026-08-11); default VLM switched to abliterated Qwen3-VL-4B; lexicon hardened (word-boundary matching, expanded `acts.yaml`). Remaining: RX 6600 throughput benchmark documented in README

---

## Current status (updated 2026-08-11)

### Phase A — ViT MVP: ✅ Complete

All 8 checklist items done. 9/9 tests pass. Committed and pushed to GitHub (`a720e44`). See "Phase A — ViT MVP" section below for the original checklist (all items checked off).

### Phase B — VLM stage: 🟡 Code complete + verified end-to-end; throughput benchmark pending

**Done 2026-08-11:**

- [x] End-to-end verification with a real video file (synthetic `testsrc2` clip) on the RX 6600: ViT path (`vnt scan` → sidecar + SQLite, GPU `cuda:0`) and VLM path (`--threshold 0 --vlm --vlm-top-k 1` → real Ollama caption in sidecar `vlm` block, `vlm_model`/`act_tags` in DB). NSFW gating confirmed: at default threshold nothing was flagged, VLM never fired.
- [x] Default VLM switched from `qwen3-vl:4b` (not actually pulled) to `hf.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M` in `config.DEFAULT_OLLAMA_MODEL`; README + `ollama.py` docstring + tests updated. `--vlm` now works out of the box against the local Ollama instance.
- [x] Lexicon hardened: `find_tags` switched from raw substring matching to word-boundary regex (`(?<!\w)…(?!\w)`) — `ass` no longer matches "glasses", `pet` no longer matches "carpet". Regression test added (`test_find_tags_uses_word_boundaries`).
- [x] `acts.yaml` expanded by user (10 → 19 tags, incl. `positions_actions`, `verbal_dirty`, `body_fluids`, `clothing_removal`, `intensity_descriptors`), then audited: 2 duplicates removed, ~70 overly generic words pruned or phrased-up (`yes`, `more`, `camera`, `master`, `plug`, `wet`, `hard`, …) to kill false positives. 522 → 449 patterns. Adversarial innocent sentence now yields 0 tags (was 4); representative NSFW caption yields 6 correct tags. 22/22 tests pass.

**Completed:**

- [x] Validate ROCm on Linux target: torch 2.9.1+rocm6.4 installed from `https://download.pytorch.org/whl/rocm6.4`; `torch.cuda.is_available()` returns `True`; device `AMD Radeon RX 6600`; `HSA_OVERRIDE_GFX_VERSION=10.3.0` confirmed working
- [x] Falconsai ViT validated on GPU: 41 ms/frame (5x faster than Mac CPU), 644 MB VRAM
- [x] Ollama VLM validated via HTTP API: `qwen3-vl:4b` runs 100% on GPU (3.5 GB VRAM, 11s per caption); 2B fallback model also works with 19% CPU offload
- [x] `pyproject.toml` updated for ROCm-only: torch pin loosened to `>=2.5`, `numpy<2` removed, `ollama>=0.4.0` + `pyyaml>=6.0` added
- [x] Existing Phase A tests pass with new ROCm environment (9/9, no regressions)
- [x] Findings documented in `docs/rocm-validation.md`

**Pending (next session):**

- [x] `ollama.py` — new module: Ollama HTTP API captioner (`OllamaCaptioner` class + module-level `caption_frames()`); default `qwen3-vl:4b`; `check_available()` fail-fast; `--vlm-model` accepts Ollama model names; NSFW-gated invocation only (frames ≥ threshold, optional `--vlm-top-k` cap by score). `vlm.py` stays as a placeholder for a future transformers backend. (done 2026-08-10)
- [x] `config.py` — added `DEFAULT_OLLAMA_MODEL` + `DEFAULT_OLLAMA_HOST`; removed config-level `DEFAULT_VLM_MODEL` (CLI `--vlm-model` now defaults to `DEFAULT_OLLAMA_MODEL`). Also fixed `DEFAULT_LEXICON` path resolution (`parents[3]` → `parents[2]`). (done 2026-08-10)
- [x] Wire `--vlm` flag into CLI scan flow; fail-fast if Ollama unreachable or model not pulled; `--unload-vit-before-vlm` flag (default off). VLM captioning runs inside the `extract.extracted_frames(...)` context. Verified end-to-end on RX 6600 (forced flagging via `--threshold 0 --vlm-top-k 1`). (done 2026-08-10)
- [x] Merge `captions[]` / `act_tags[]` into sidecar `vlm` block (with `backend`/`host`); DB gets `vlm_model` + `act_tags` only; non-VLM re-scan preserves existing VLM data (decision #18, verified). (done 2026-08-10)
- [x] Phase B unit/integration tests (`tests/test_ollama.py` with mocked `ollama.Client`; integration test skips when Ollama unavailable). Frame-selection extracted as pure `select_flagged_frames()` helper. 24/24 tests pass. (done 2026-08-10)
- [ ] Throughput benchmark on RX 6600 (ViT path + gated VLM path); document in README
- [x] Rich progress for VLM stage: per-frame caption progress ("captioning frame 3/7") — at ~11s/caption a bare per-video spinner is too opaque. Implemented in `cli.py` ("Captioning {video} frame {i}/{n}").

### Scope changes from original plan

| # | Original decision | Changed to | Reason |
| --- | ------------------- | ------------ | -------- |
| 6 | VLM model: `Qwen/Qwen2.5-VL-3B-Instruct` via transformers | `qwen3-vl:4b` via **Ollama HTTP API** | Ollama manages VRAM/quantization; 4B q4_K_M (3.3 GB) runs 100% on GPU vs 3B fp16 (6 GB) that wouldn't fit alongside other services. No need for `qwen-vl-utils`/`accelerate` in the Python project. |
| 9 | Packaging: PDM | setuptools (kept `[tool.pdm]` for scripts only) | Deviation from Phase A; build backend is `setuptools.build_meta` with src layout. PDM scripts retained via `[tool.pdm.scripts]`. |
| — | Mac dev constraint (torch<2.3, numpy<2) | ROCm-only target | Project moved to ROCm server. torch pin loosened to `>=2.5` for ROCm wheels (2.5–2.10). Mac CPU dev path no longer supported. |
| — | VLM loaded directly via transformers | VLM called via Ollama HTTP API (`localhost:11434`) | Ollama container (`ollama-rocm`) already running with ROCm. Shared model store via bind-mount of `~/.ollama`. Avoids VRAM management complexity. Models: `qwen3-vl:4b` (default), 2B abliterated (fallback). |
| — | VLM backend in `vlm.py` | **Separate `ollama.py` module**; `vlm.py` stays as placeholder | `vlm.py` remains reserved for a future transformers-based VLM backend. `ollama.py` is the first concrete backend (Ollama HTTP API). Later, `vlm.py` can become a dispatcher routing to `ollama.py` or a transformers backend. Keeps backends pluggable without entangling Ollama-specific code with the transformers placeholder. |
| 6b | Default VLM: `qwen3-vl:4b` | `hf.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M` | `qwen3-vl:4b` was validated during Phase B but never actually pulled into the shared `~/.ollama` store; the abliterated Huihui Qwen3-VL-4B (Q4_K_M, 3.0 GB) is what's deployed. Abliteration also mitigates NSFW refusals. `--vlm` works out of the box now (done 2026-08-11). |
| 7b | Lexicon matching: case-insensitive substring | **Word-boundary regex** (`(?<!\w)…(?!\w)`) | Substring matching caused false positives (`ass` in "glasses", `pet` in "carpet"). Switch plus pruning of ~70 generic words in `acts.yaml` reduced adversarial-sentence false positives from 4 tags to 0 (done 2026-08-11). |

### Environment details (ROCm server)

- **GPU:** AMD Radeon RX 6600 (gfx1032, 8 GB VRAM)
- **ROCm:** 7.1.1; `HSA_OVERRIDE_GFX_VERSION=10.3.0` required for gfx1032
- **Python:** 3.12.13 in `.venv/`
- **PyTorch:** 2.9.1+rocm6.4 (from `https://download.pytorch.org/whl/rocm6.4`)
- **Ollama:** 0.32.5 (Docker container `ollama-rocm`, bind-mount `~/.ollama`)
- **VLM models available:** `qwen3-vl:4b` (3.3 GB, 100% GPU), `hf.co/lihaoyun6/Qwen3-VL-2B-Instruct-abliterated_GGUF:Q5_K_M` (2.1 GB, 81% GPU), `hf.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M` (3.0 GB)
- **Full validation report:** `docs/rocm-validation.md`

---

## Locked decisions

| # | Decision | Choice |
| --- | ---------- | -------- |
| 1 | v1 feature set | Broad NSFW gate + **VLM captions** (act descriptions second stage) |
| 2 | Primary target | **Linux + RX 6600 (gfx1032) + ROCm**; Intel Mac for light CPU smoke tests only |
| 3 | Output | **JSON sidecar per video + SQLite index** |
| 4 | SQLite location | `./video_nsfw_index.db` in cwd / next to first input; `--db PATH` override |
| 5 | VLM invocation | **NSFW-gated only** — VLM runs solely on frames above the NSFW threshold (plus optional top-K cap) |
| 6 | VLM model | **Pluggable: default `hf.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M` (abliterated Qwen3-VL 4B) via Ollama HTTP API** (was `qwen3-vl:4b`, and before that `Qwen/Qwen2.5-VL-3B-Instruct` via transformers — see scope changes), override via `--vlm-model` |
| 7 | Act-tag derivation | **Keyword/phrase lexicon parse** of captions (deterministic, editable, no extra model); matching is case-insensitive with word boundaries (not raw substring) |
| 8 | Interface | **CLI only** for v1 (Gradio deferred) |
| 9 | Packaging | **setuptools** (`pyproject.toml`, src layout, console script entrypoint); `[tool.pdm]` retained for scripts only (was PDM — changed during Phase A) |
| 10 | Phasing | **Phase A: ViT MVP → Phase B: VLM stage** (validate ROCm before loading the 3B VLM) |
| 11 | VLM backend module | **`ollama.py`** (Ollama HTTP API) is the first concrete backend; `vlm.py` stays as a placeholder for a future transformers backend. CLI calls `ollama.py` directly when `--vlm` is set. |
| 12 | VLM prompt | **Targeted description** — e.g. "Describe what is happening in this image: the people, their actions, state of dress, and the setting." Produces richer captions that match more lexicon patterns. Stored as a module constant, easy to tune. |
| 13 | Ollama unavailable | **Fail fast** — when `--vlm` is passed but Ollama is unreachable or the model isn't pulled, exit 1 with a clear message before scanning starts. No sidecar written. |
| 14 | VRAM management | **Configurable: `--unload-vit-before-vlm` flag** (default off). Off = keep both loaded (~4.1 GB peak, faster for batch). On = `del pipe` + `torch.cuda.empty_cache()` before VLM calls, reload ViT for next video (~3.5 GB peak, safer when other GPU services are running). |
| 15 | Ollama `keep_alive` | **`keep_alive="10m"` during scans, explicit unload at scan end.** Prevents Ollama's default 5-min unload from forcing a mid-scan model reload (4s load penalty per caption), and frees VRAM for other services when the scan finishes. |
| 16 | Mid-scan VLM failure | **Degrade, don't abort.** If captioning fails on video N of a batch (timeout/OOM/crash), log a warning, write the sidecar with ViT results and `vlm: null`, and continue to the next video. Decision #13 fail-fast applies only to the pre-scan availability check — ViT results remain valuable even when the VLM stage fails. |
| 17 | VLM request timeout | **120s default per caption request** on `ollama.Client`. At ~11s/caption a hung request without a timeout stalls a batch scan indefinitely. |
| 18 | Re-scan idempotency | **Non-VLM re-scan preserves existing VLM data.** Upsert only touches columns the current scan produced: a scan without `--vlm` leaves `vlm_model`/`act_tags` (DB) and the sidecar `vlm` block untouched; a scan with `--vlm` overwrites them. Prevents accidental clobbering of expensive caption results. |

## Objective

Ship `video-nsfw-tagger` as an installable CLI:

- **Phase A (MVP):** `vnt scan <path>` — ffmpeg frame sampling at 1 fps → Falconsai binary NSFW score per frame → aggregate (`nsfw_percent`, `max_score`, `flagged_frames[]`, `verdict` at configurable threshold) → JSON sidecar (`video.mp4` → `video.nsfw.json`) + upsert into `./video_nsfw_index.db`. Runs on CPU (Mac dev) and ROCm (target).
- **Phase B (VLM):** opt-in `--vlm` flag — for frames above threshold, caption with local VLM → parse captions against a lexicon file → merge `captions[]` and `act_tags[]` into sidecar + DB.

## Feasibility findings (verified, prior session)

| Component | Result | Evidence |
| ----------- | -------- | ---------- |
| ffmpeg 8.1.2 frame extraction @ 1 fps | **Pass** | 3s clip → 3 PNGs in ~20 ms |
| `Falconsai/nsfw_image_detection` (ViT, Apache-2.0, ~327 MB) | **Pass** | CPU pipeline; synthetic frame scored `normal` 0.9996; ~0.5 s first frame, ~0.21 s/frame batched |
| CLIP zero-shot (`open_clip` ViT-B-32) | **Weak** | High NSFW mass on SFW test pattern → unreliable alone; not in v1 path |
| `Qwen/Qwen2.5-VL-3B-Instruct` | **Superseded** | Originally planned for direct transformers loading. Replaced by `qwen3-vl:4b` via Ollama HTTP API during Phase B validation — see scope changes above. |
| `qwen3-vl:4b` via Ollama (Phase B validation) | **Pass** | 100% GPU on RX 6600, 3.5 GB VRAM, 11s per caption (4s load + 4.7s eval). Accurate captions on synthetic test pattern. |
| End-to-end scan with real video (2026-08-11) | **Pass** | Synthetic 5s `testsrc2` clip: ViT path wrote sidecar + DB on GPU; forced-flag VLM path (`--threshold 0 --vlm-top-k 1`) produced a real caption from the abliterated Qwen3-VL-4B; gating confirmed (no VLM call at default threshold). |
| Abliterated Qwen3-VL-4B (`Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M`, now default) | **Pass** | Pulled in shared `~/.ollama`; passes `check_available()` fail-fast and captions real frames via `vnt scan --vlm` with no `--vlm-model` override. |
| Offline after first download | **Yes** | ViT caches under `~/.cache/huggingface`; VLM models in `~/.ollama` (shared with Docker container) |

**Throughput (validated on RX 6600/ROCm):** ViT 41 ms/frame (5x faster than Mac CPU's 210 ms/frame). VLM 11s per caption with `qwen3-vl:4b`.

## Architecture

```dir
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
├── vlm.py            # Phase B placeholder (future transformers backend; untouched in this iteration)
├── ollama.py         # Phase B: Ollama HTTP API captioner (abliterated Qwen3-VL-4B default) — first concrete VLM backend
└── lexicon.py        # Phase B: load lexicon YAML/JSON, caption → act_tags (word-boundary matching)
lexicon/
└── acts.yaml         # editable keyword/phrase → tag mapping (19 tags, 449 patterns)
tests/
├── test_aggregate.py
├── test_sidecar.py
├── test_db.py
├── test_lexicon.py   # Phase B (incl. word-boundary regression test)
├── test_ollama.py    # Phase B (mocked ollama.Client + optional integration)
└── fixtures/         # synthetic ffmpeg-generated clips (no real content)
pyproject.toml        # setuptools backend, src layout; [project.scripts] vnt = "video_nsfw_tagger.cli:app"; [tool.pdm.scripts] retained for lint/test shortcuts
README.md
```

## Appendix: original `pyproject.toml` boilerplate (historical)

The project was bootstrapped from this PDM/Typer/Ruff boilerplate during Phase A. Kept for reference only — the actual `pyproject.toml` in the repo is authoritative (setuptools backend, ROCm pins).

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
    "backend": "ollama",
    "model": "qwen3-vl:4b",
    "host": "http://localhost:11434",
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
         [--vlm] [--vlm-model MODEL] [--vlm-top-k N] [--lexicon PATH]
         [--unload-vit-before-vlm]
vnt report [--db PATH] [--verdict nsfw] [--min-percent 10]
```

- `scan` on a directory batches all supported video files; writes sidecar next to each video and upserts into the DB (default `./video_nsfw_index.db`). Rich console for progress output.
- `--vlm` enables VLM captioning via Ollama (Phase B). Fails fast if Ollama is unreachable or the model isn't pulled. `--vlm-model` accepts Ollama model names (e.g. `qwen3-vl:4b`, `hf.co/lihaoyun6/Qwen3-VL-2B-Instruct-abliterated_GGUF:Q5_K_M`).
- `--unload-vit-before-vlm` (default off): when set, the ViT pipeline is unloaded (`del` + `torch.cuda.empty_cache()`) before VLM calls on each video, then reloaded for the next. Use when VRAM is constrained (other GPU services running). Off keeps both loaded (~4.1 GB peak, faster for batch scans).

## Implementation plan

### Phase A — ViT MVP (develop and verify on this Mac; no VLM deps) — ✅ Complete

- [x] `pdm init` project: Python 3.12 requires-python, deps pinned for **Mac Phase A** (`typer[all]`, `rich`, `torch==2.2.2`, `transformers>=4.40,<5`, `numpy<2`, `pillow`, `safetensors` — no VLM/Qwen/accelerate-VLM, no `torch>=2.4`); console script `vnt` and `python -m video_nsfw_tagger` entry points
- [x] `device.py` — resolve `auto`: cuda→mps→cpu; expose torch/ROCm info for `vnt config-show`
- [x] `extract.py` — ffprobe duration; ffmpeg `-vf fps=1` into `tempfile.TemporaryDirectory`; yield (index, timestamp, path); guarantee cleanup
- [x] `classify.py` — lazy-load pipeline once; batch inference (batch size flag, default 8)
- [x] `aggregate.py` + `sidecar.py` — stats, verdict, atomic sidecar write
- [x] `db.py` — stdlib sqlite3, idempotent schema, upsert by `path`
- [x] `cli.py` — `scan` (file/dir/recursive), `report`, `config-show`
- [x] Tests: pure-logic units (aggregate, sidecar round-trip, db upsert) + one synthetic ffmpeg clip integration test; no network, no real content
- [x] README: install (PDM), quickstart, threshold tuning notes, privacy statement

**Phase A acceptance criteria:**

- `vnt scan` on a synthetic clip produces a valid sidecar matching the schema and a DB row; exit 0 on CPU-only Mac.
- Re-scanning the same file updates (not duplicates) the DB row.
- Temp frames are removed after every run (success and failure paths).
- Unit + integration tests pass offline (`HF_HUB_OFFLINE=1` after first model download).

### Phase B — VLM stage (develop against ROCm target; CPU optional/slow) — 🟡 Code complete + verified end-to-end (2026-08-11); only README throughput benchmark remains

- [x] Validate ROCm on Linux target: torch 2.9.1+rocm6.4 from `https://download.pytorch.org/whl/rocm6.4`; `torch.cuda.is_available()` returns `True`; `HSA_OVERRIDE_GFX_VERSION=10.3.0` confirmed; `rocm-smi` shows GPU during load. ViT: 41 ms/frame, 644 MB VRAM. Ollama VLM: `qwen3-vl:4b` 100% GPU, 3.5 GB VRAM, 11s/caption.
- [x] `ollama.py` — new module: `OllamaCaptioner` class (wraps `ollama.Client`) + module-level `caption_frames()`; default `qwen3-vl:4b`; `check_available()` fail-fast (verifies Ollama reachable + model pulled); targeted prompt (module constant); `keep_alive="10m"` during scans + explicit unload at scan end (decision #15); 120s per-request timeout (decision #17); NSFW-gated invocation only (frames ≥ threshold, optional `--vlm-top-k` cap by score, via a pure frame-selection helper for testability). `vlm.py` stays untouched as a placeholder for a future transformers backend.
- [x] `config.py` — add `DEFAULT_OLLAMA_MODEL` + `DEFAULT_OLLAMA_HOST = "http://localhost:11434"`; **remove the config-level `DEFAULT_VLM_MODEL`** (duplicated in `vlm.py`; currently also feeds the CLI `--vlm-model` default — three sources of truth). CLI `--vlm-model` default switches to `DEFAULT_OLLAMA_MODEL`. Default model later set to the abliterated Qwen3-VL-4B GGUF actually pulled on the server (2026-08-11).
- [x] `lexicon.py` + `lexicon/acts.yaml` — keyword/phrase → tag mapping; case-insensitive **word-boundary** match (hardened from substring matching 2026-08-11); editable without code changes (implemented in Phase A; `acts.yaml` expanded to 19 tags / 449 patterns and pruned of generic false-positive words 2026-08-11)
- [x] Wire `--vlm` into CLI scan flow: fail-fast check before scan loop; per-video VLM captioning on flagged frames **inside the `extract.extracted_frames(...)` context** (frames are deleted on exit — captioning after the `with` block would hit missing files); `--unload-vit-before-vlm` flag (default off); remove Phase A no-op notice; per-frame caption progress output
- [x] Merge `captions[]` / `act_tags[]` into sidecar `vlm` block (with `backend` + `host` fields); DB gets `vlm_model` + `act_tags` only — captions stay sidecar-only, existing columns suffice, no migration needed
- [x] `tests/test_ollama.py` — unit tests with mocked `ollama.Client` (top_k selection, frame-to-path mapping, caption structure, `check_available` success/failure); integration test marked `@pytest.mark.integration` (skips when Ollama unavailable or model missing)
- [ ] Throughput benchmark on RX 6600 (ViT path and gated VLM path); document results in README

**Phase B acceptance criteria:**

- `vnt scan --vlm` on a flagged clip adds non-empty `vlm.captions` to the sidecar and lexicon-derived `act_tags` where matched. **Verified 2026-08-11** with a real video file (forced flagging via `--threshold 0`) — sidecar `vlm` block populated, DB row carries `vlm_model`/`act_tags`.
- `vnt scan --vlm` fails fast with a clear error if Ollama is unreachable or the model isn't pulled (no sidecar written).
- Mid-scan VLM failure degrades gracefully: sidecar written with ViT results + `vlm: null`, batch continues (decision #16).
- Non-VLM re-scan of a previously VLM-scanned video preserves existing `vlm` sidecar block and DB `vlm_model`/`act_tags` (decision #18).
- VLM never runs on frames below the NSFW threshold (verified by log/call count in tests).
- `--unload-vit-before-vlm` peak VRAM verified by measurement, not assertion: integration test captures `torch.cuda.max_memory_allocated()` (or `rocm-smi` sample) for both flag states; unloaded ≈3.5 GB, loaded ≈4.1 GB.
- `vlm.py` remains untouched (placeholder for future transformers backend).
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
| ------ | ----------- | -------- | ------------ |
| ~~ROCm/PyTorch on gfx1032~~ | ~~Medium~~ | ~~High~~ | **Resolved:** torch 2.9.1+rocm6.4 works with `HSA_OVERRIDE_GFX_VERSION=10.3.0`; ViT and VLM both validated on GPU |
| ~~VLM too heavy for 8 GB VRAM~~ | ~~Medium~~ | ~~Medium~~ | **Resolved:** Ollama manages VRAM; `qwen3-vl:4b` (q4_K_M, 3.3 GB) runs 100% on GPU with 3.5 GB VRAM; 2B fallback available with CPU offload |
| Ollama VLM refuses NSFW content | Low | Medium | **Mitigated:** default is the abliterated `Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M` (2026-08-11); 2B fallback is also abliterated. Documented in README. |
| Act-tag false positives from lexicon | ~~High~~ Low | Low–Med | **Reduced 2026-08-11:** word-boundary matching in `find_tags` + pruning of ~70 generic words in `acts.yaml`; adversarial innocent text yields 0 tags. Lexicon remains deterministic, auditable, editable; captions stored alongside tags for user review |
| Long-video runtime | Medium | Medium | 1 fps sampling, batch inference, `--max-duration`, GPU path |
| Model bias / accuracy unknown on real corpora | Medium | Medium | Only smoke-tested; user tunes `--threshold` on own samples; document limitation |

## Verification commands

```bash
# Setup (ROCm target)
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install -e ".[dev]"
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # required for RX 6600 (gfx1032)

vnt config-show                    # torch version, device resolution

# Synthetic end-to-end (offline-safe after first model download)
ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=30 -pix_fmt yuv420p -y /tmp/vnt-sample.mp4
vnt scan /tmp/vnt-sample.mp4
cat /tmp/vnt-sample.nsfw.json
vnt report

# Tests
pytest -q

# ROCm / Ollama validation
rocm-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
docker exec ollama-rocm ollama list    # check VLM models available
vnt scan --vlm /path/to/sample.mp4     # Phase B; add --threshold 0 --vlm-top-k 1 on a benign clip to force the VLM path
```

## Notes

- All processing local; the "decline if minor-related" rule is a hard boundary outside the pipeline.
- Test fixtures must be synthetic (ffmpeg-generated patterns) — no real NSFW/SFW content in the repo or tests.
- ViT model (~327 MB) caches under `~/.cache/huggingface`; first run downloads it.
- VLM models are stored in `~/.ollama` (shared with the `ollama-rocm` Docker container via bind-mount). The default is the abliterated Qwen3-VL-4B GGUF (3.0 GB), already pulled; `qwen3-vl:4b` (3.3 GB, validated during Phase B) would need `docker exec ollama-rocm ollama pull qwen3-vl:4b`.
- Abliterated VLM models are used because the base Qwen3-VL may refuse to caption NSFW content. This is documented in the README and `docs/rocm-validation.md`.
- Follow the repo's writing-style guide for README/docs (warm, clear, direct, practical).
