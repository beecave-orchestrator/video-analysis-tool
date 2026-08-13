# VLM Prompt Experiments

How to test different captioning prompts against the VLM and pick the one
that produces the best act-tag classification for your model.

The default VLM is
`hf.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M`,
but this workflow applies to any Ollama vision model (`--vlm-model`).

## Why

Tag derivation works in two steps:

1. The VLM writes a free-text caption for each flagged frame.
2. `lexicon/acts.yaml` keywords are regex-matched against that caption.

So tag quality depends entirely on the words the VLM happens to use. A
prompt that makes the model write "naked" instead of "undressed", or
"doggystyle" instead of "from behind", directly changes which tags fire.
The prompt experiment measures this: same frames, same lexicon, 10
different prompts.

## Components

| Component | Purpose |
| --------- | ------- |
| `experiments/prompts/*.md` | Candidate prompts, one file each; the text under the `## Prompt` heading is what gets sent to the VLM |
| `scripts/10-prompt-test.sh` | Driver: loops over the prompts, runs `vnt scan` for each, collects results |
| `--frame-cache DIR` | `vnt scan` option: caches extracted frames + ViT scores so only the (slow) VLM stage re-runs per prompt |
| `--vlm-prompt-file` / `--vlm-prompt` / `--vlm-prompt-id` | `vnt scan` options: override the captioning prompt and label the run |
| `vnt prompt-report RUN_DIR` | Aggregates all runs into a comparison table + JSON + Markdown report |

## Quick start

```bash
# From the repo root, with Ollama running:
scripts/10-prompt-test.sh watch/sample.mp4 --top-k 10
```

The script finds `vnt` automatically: it uses an activated venv if present,
falls back to `.venv/bin/vnt`, and finally to `pdm run vnt`. If none are
available it exits with an explanatory error.

Results land in `experiments/results/<YYYYMMDD-HHMMSS>/`:

```text
experiments/results/20260812-234500/
├── prompts/                  # copy of the prompt set used (reproducibility)
├── index.db                  # SQLite index for all runs in this experiment
├── 01_baseline/
│   ├── sample.mp4.nsfw.json  # sidecar with captions + tags for this prompt
│   └── .prompt.txt           # the exact prompt text that was sent
├── 02_minimal/
├── ...
├── report.json               # machine-readable metrics for all prompts
└── report.md                 # human-readable report incl. caption dumps
```

The first prompt's run performs frame extraction and ViT classification
and stores them in `experiments/cache/`; the other nine runs reuse that
cache and go straight to captioning.

## Options

```text
scripts/10-prompt-test.sh TARGET [OPTIONS]

  --top-k N          Flagged frames captioned per prompt (default 10)
  --fps F            Frame sampling rate (default 1.0)
  --prompts-dir DIR  Prompt set (default experiments/prompts)
  --results-dir DIR  Results root (default experiments/results)
  --cache-dir DIR    Frame/score cache (default experiments/cache)
  --vlm-timeout SEC  Per-caption timeout (default 600; cold load counts)
  --extra-opts STR   Extra options forwarded to vnt scan, e.g. "--recursive"
```

The script applies several latency mitigations automatically:

- **Thinking disabled** — `vnt scan` passes `think=False` to Ollama by
  default, so the "Thinking" model variant answers directly instead of
  burning minutes on hidden reasoning (opt back in with
  `--extra-opts "--vlm-think"`).
- **Model kept warm** — `--keep-vlm-loaded` is passed for all but the
  last prompt, so the model is loaded once per experiment instead of
  once per prompt.
- **One retry per frame** — a timed-out caption is retried once before
  counting as a failure (`--vlm-retries`, default 1).
- **VRAM preflight** — the script warns when `rocm-smi` shows >60 % VRAM
  already in use (other GPU containers can force partial CPU offload,
  which turns ~40 s captions into multi-minute timeouts), and passes
  `--unload-vit-before-vlm` to free the ViT during captioning.

Always smoke-test first with `--top-k 2` (10 prompts × 2 captions ≈
20 VLM calls) before a full run. The frame cache makes re-runs with a
different `--top-k` or prompt set cheap, and fully cached runs skip the
ViT load entirely.

## The 10 candidate prompts

| # | File | Strategy |
| - | ---- | -------- |
| 01 | `01_baseline.md` | Current default prompt (control) |
| 02 | `02_minimal.md` | Bare "Describe this image." |
| 03 | `03_anatomical.md` | Clinical/anatomical vocabulary |
| 04 | `04_explicit_vocab.md` | Explicit terms that mirror the lexicon |
| 05 | `05_structured_fields.md` | Fixed fields: People/Clothing/Acts/Positions/Objects/Setting |
| 06 | `06_lexicon_aware.md` | Names the 19 lexicon categories in the prompt |
| 07 | `07_json_output.md` | Constrained JSON output |
| 08 | `08_roleplay_directive.md` | "Adult-content metadata tagger" persona |
| 09 | `09_step_by_step.md` | Chain-of-thought structure (suits Thinking models) |
| 10 | `10_negative_space.md` | Also state what is NOT happening |

## Reading the report

`vnt prompt-report` prints a Rich table and writes `report.json` +
`report.md` into the run directory. Per prompt you get:

- **attempted / success / failures** — caption calls made, non-empty
  captions returned, and Ollama failures.
- **tagged%** — share of captioned frames that produced at least one
  lexicon tag.
- **distinct tags** — how many of the 19 lexicon categories fired at
  least once across the run.
- **per-tag pattern counts** — how many keyword patterns matched per
  category (one column per lexicon tag).
- **avg chars / avg s** — caption length and caption wall time (Thinking
  prompts can be dramatically slower; this is real cost).
- **Caption dumps** (in `report.md`) — every caption with its matched
  tags, for manual eyeballing.

## How to pick the best prompt

The table is ranked by `distinct tags`, then `tagged%` — a prompt that
names more lexicon categories is ranked higher. Treat this as a starting
point, not a verdict:

1. **Coverage first.** More distinct tags = the model's vocabulary
   overlaps more of the lexicon. If 01_baseline hits 5 tags and
   04_explicit_vocab hits 12, the latter is probably the better default.
2. **Check precision by eye.** High hit counts can mean hallucinated
   acts. Read the caption dumps in `report.md`: are the tagged acts
   actually visible in the frame? A prompt that "wins" by claiming
   everything is a gangbang is worse than one with fewer, accurate hits.
3. **Check latency.** If two prompts tie on coverage, prefer the one
   with lower `avg s` — Thinking prompts that take 2–3× longer rarely
   pay off.
4. **Check failures.** Non-zero `failures` or empty captions can
   indicate the prompt confuses the model (common with 07_json_output).
5. **Confirm on a second video.** Re-run the winner on different
   content before making it the default prompt.

Once chosen, set the default in `src/video_nsfw_tagger/ollama.py`
(`DEFAULT_PROMPT`) or always pass it via `--vlm-prompt-file`.

## Writing your own prompt

Add a file to `experiments/prompts/` (or any `--prompts-dir`):

```markdown
# 11 My Prompt

## Goal

One line on what hypothesis this prompt tests.

## Prompt

The exact text sent to the VLM goes here.
```

The script extracts everything under `## Prompt`; if that heading is
missing it sends the whole file. Prompts run in filename order, so the
`NN_` prefix controls ranking display.

Tips for this pipeline:

- The matcher is case-insensitive but needs whole-word matches —
  synonyms the lexicon doesn't contain never fire. Check
  `lexicon/acts.yaml` before writing a prompt.
- Very long instructions can crowd out actual image description on a 4B
  model; watch caption length and content drift in the dumps.
- Structured output (07) only helps if the model reliably returns valid
  JSON; the report falls back to raw-text matching, but a broken JSON
  caption usually means fewer tag hits anyway.

## Troubleshooting

| Symptom | Likely cause |
| ------- | ------------ |
| Script fails at first prompt | Ollama not running or model not pulled — `vnt scan` fails fast; check `ollama list` |
| Second run re-extracts frames | The video file changed (mtime/size) — the cache key covers content, so edits invalidate it |
| Sidecar overwritten / old results gone | Sidecars live next to the video and are overwritten per scan; the script copies each run's sidecars into the run dir, so only sidecars outside experiment runs are affected |
| 07_json_output scores zero | The model didn't return JSON or wrapped it in markdown fences — check the caption dumps; the fallback matcher only sees raw text |
| Everything is slow | Reduce `--top-k`, or pre-warm: the first VLM call includes model load time and counts against `--vlm-timeout` (default 300 s) |
