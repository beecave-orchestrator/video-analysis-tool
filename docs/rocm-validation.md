# ROCm Validation Notes — Phase B Environment

Validation of the Phase B environment on the AMD RX 6600 / ROCm server.
Performed 2026-08-10.

## Hardware & software

| Component | Value |
| --------- | ----- |
| GPU | AMD Radeon RX 6600 (gfx1032, 8 GB VRAM) |
| ROCm | 7.1.1 |
| OS | Linux 6.8.0-136-generic (Ubuntu 22.04) |
| Python | 3.12.13 |
| PyTorch | 2.9.1+rocm6.4 (installed from `https://download.pytorch.org/whl/rocm6.4`) |
| torchvision | 0.24.1+rocm6.4 |
| transformers | 4.57.6 |
| Ollama | 0.32.5 (Docker container `ollama-rocm`, ROCm image) |
| ffmpeg | 4.4.2 |

## gfx1032 workaround

The RX 6600 reports as gfx1032, which ROCm does not officially support for
compute. Setting `HSA_OVERRIDE_GFX_VERSION=10.3.0` makes ROCm treat it as
gfx1030 (RX 6800), which is supported. This is set in the Ollama container
environment and must be exported before running `vnt`:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

## Environment setup

```bash
cd /home/elvee/Local-AI/video-analysis-tool
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# ROCm torch (4.5 GB download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# Project deps (torch already satisfied)
pip install -e ".[dev]"
```

## torch.cuda verification

```text
torch 2.9.1+rocm6.4
cuda True
hip 6.4.43484-123eb5128
device AMD Radeon RX 6600
```

`vnt config-show` resolves auto device to `cuda:0`.

## Falconsai ViT on GPU

Model: `Falconsai/nsfw_image_detection` (~327 MB download on first run).

| Metric | Value |
| ------ | ----- |
| Model load time (cold, includes download) | 33.4 s |
| First inference (includes kernel compilation) | 3.7 s |
| Batch of 5 frames (warm) | 0.21 s total, 41.2 ms/frame |
| VRAM usage (model loaded) | ~644 MB |

Synthetic test pattern scored `normal` 0.9997, `nsfw` 0.0003 — correct.

**Speedup vs Mac CPU:** ~5x (41 ms/frame vs ~210 ms/frame on i5-8257U).

## Ollama VLM models

Three Qwen3-VL models are available in the local Ollama store
(`~/.ollama`, bind-mounted into the container):

| Model | Size | Quant | VRAM | GPU/CPU split | Load time | Eval (246 tok) |
| ----- | ---- | ----- | ---- | ------------- | --------- | -------------- |
| `hf.co/lihaoyun6/Qwen3-VL-2B-Instruct-abliterated_GGUF:Q5_K_M` | 2.1 GB | Q5_K_M | ~1.6 GB | 81%/19% | 6.7 s | 2.4 s (28 tok) |
| `qwen3-vl:4b` | 3.3 GB | Q4_K_M | 3.5 GB | 100% GPU | 4.0 s | 4.7 s (246 tok) |
| `hf.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF:Q4_K_M` | 3.0 GB | Q4_K_M | — | — | — | — |

**Recommended default for Phase B:** `qwen3-vl:4b` — runs 100% on GPU with
3.5 GB VRAM, best caption quality, and fits comfortably alongside the ViT
(644 MB) within the 8 GB budget.

The 2B model is a lighter fallback when VRAM is constrained (e.g. when other
GPU services are running). It uses 19% CPU offload but still produces accurate
captions.

### VLM API call example

```python
import base64, requests

with open("frame.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:11434/api/chat", json={
    "model": "qwen3-vl:4b",
    "messages": [{"role": "user", "content": "Describe this image.", "images": [img_b64]}],
    "stream": False,
})
caption = resp.json()["message"]["content"]
```

### Caption quality (synthetic test pattern)

- **2B:** "A circular television test pattern with vertical stripes of rainbow
  colors and a horizontal spectrum bar, featuring a small black rectangle on
  the right side."
- **4B:** "This image is a classic television test pattern featuring vertical
  color bars (cyan, magenta, blue, yellow, green, red) within a circular
  frame, a horizontal rainbow gradient bar across the center, and a pixelated
  '8' symbol in the top-right corner."

Both correctly identify the synthetic test pattern. The 4B model provides more
detail (names individual colors, spots the pixelated symbol).

## Ollama container configuration

The `ollama-rocm` container is configured via
`/home/elvee/Local-AI/ollama/docker-compose.yaml`:

- Bind-mounts `~/.ollama` to `/root/.ollama` (shared model store with host)
- ROCm device access (`/dev/kfd`, `/dev/dri`)
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` for gfx1032 workaround
- API at `http://localhost:11434`

## VRAM budget

| Scenario | VRAM used | Notes |
| -------- | --------- | ----- |
| Idle (no services) | ~62 MB | Clean GPU |
| ViT loaded | ~644 MB | Falconsai ViT only |
| 4B VLM loaded | 3.5 GB | 100% GPU, no offload |
| ViT + 4B VLM (sequential) | ~4.1 GB | Run one at a time; Ollama unloads after 5 min idle |
| With parakeet + whisper services | ~5-6 GB | Other services must be stopped for full-GPU VLM |

**Recommendation:** In the pipeline, run ViT first (score all frames), then
unload ViT, then call Ollama VLM on flagged frames only. This keeps peak VRAM
under 4 GB and avoids conflicts with other GPU services.

## Test suite

```text
9 passed in 0.47s
```

All Phase A tests pass with the new ROCm torch environment. No regressions.

## Notes for Phase B implementation

1. **VLM via Ollama HTTP API** (not direct transformers loading) — Ollama
   manages VRAM, quantization, and model lifecycle. The `ollama` Python
   package (v0.6.2) is already installed.
2. **Default VLM model** should be `qwen3-vl:4b` (not `Qwen/Qwen2.5-VL-3B-
   Instruct` as in the original megaplan) — it's already in the local Ollama
   store, runs 100% on GPU, and produces better captions.
3. **`--vlm-model` override** should accept Ollama model names (e.g.
   `qwen3-vl:4b`, `hf.co/lihaoyun6/Qwen3-VL-2B-Instruct-abliterated_GGUF:Q5_K_M`).
4. **Abliterated models** are used because the base Qwen3-VL may refuse to
   caption NSFW content. The abliterated versions have safety filtering
   removed. Document this clearly in the README.
5. **`HSA_OVERRIDE_GFX_VERSION=10.3.0`** must be set before running `vnt` on
   this server. Consider adding it to the venv activation script or
   documenting it prominently.
