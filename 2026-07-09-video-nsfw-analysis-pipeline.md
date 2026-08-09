# Video NSFW Analysis Pipeline — Local Offline Tagging

- **Created:** 09-07-2026 23:55 Europe/Amsterdam
- **Status:** planned
- **Owner:** investigator

## Goal

Implement and verify a local, offline, privacy-first pipeline for automating analysis of video files to tag them if they contain NSFW/explicit content or specific sexual acts, running on AMD RX 6600 (gfx1032) with ROCm-enabled PyTorch where possible, fallback to CPU/ONNX. Everything stays on the user's machine; no uploads.

## Current state

- Hardware: AMD RX 6600 / gfx1032, Linux, HSA_OVERRIDE_GFX_VERSION=gfx1030 available.
- Recommended models identified: Falconsai/nsfw_image_detection (ViT), CLIP zero-shot, potential VLM chaining (Qwen2.5-VL), FlowNSFW for temporal.
- Frame extraction via ffmpeg or OpenCV is standard and fast.
- No existing pipeline script or to-do in agent-output for this exact workflow.
- General NSFW image models exist and are lightweight; video requires frame sampling + aggregation.

## Context

The user requested a way to automate analysis of video files to tag them for specific sexual acts or general NSFW. Public tools support broad NSFW detection via frame classification; fine-grained "specific act" detection requires custom prompts or VLM chaining and is less reliable without private fine-tuning. The pasted practical approach provides a clear, verifiable starting point using open-source components (ffmpeg, transformers, HF models). This to-do tracks the implementation, GPU tuning, testing, and tagging logic while respecting privacy and the "decline if minor-related" rule.

## Scope

**In scope:**
- Local offline pipeline: ffmpeg/OpenCV frame extraction (1 fps sampling).
- Classification with HF NSFW ViT (Falconsai recommended) or CLIP zero-shot with custom prompts.
- Tagging logic: confidence threshold → JSON sidecar / filename prefix / DB entry; log flagged frames.
- GPU setup: ROCm PyTorch for inference on RX 6600; fallback CPU/ONNX.
- Enhancements: optical flow (FlowNSFW), VLM chaining for act descriptions, Gradio UI or CLI.
- Verification on sample videos; accuracy testing and bias notes.
- Documentation of commands, model loading, and rollback.

**Out of scope:**
- Fine-tuning models on private datasets.
- Cloud APIs or any data upload.
- Handling of minor-related content (explicit decline per rules).
- Production deployment or legal compliance review (user responsibility for EU AI Act etc.).
- Specific sexual act datasets or high-accuracy act classifiers (not publicly available).

## Plan

- [ ] **Investigation:** Confirm ROCm/PyTorch setup on RX 6600; test basic model load (Falconsai/nsfw_image_detection) and ffmpeg frame extraction.
- [ ] **Checkpoint 1:** Create to-do, commit and push.
- [ ] **Fix/execution:**
  - [ ] Implement frame extraction script (ffmpeg subprocess or ffmpeg-python, 1 fps sampling, save to temp dir).
  - [ ] Build classification module: load pipeline or AutoModelForImageClassification for Falconsai; alternative CLIP zero-shot with prompts for NSFW/specific acts.
  - [ ] Implement aggregation/tagging: per-video % NSFW, flagged frames list, write JSON sidecar or exiftool tags or SQLite entry.
  - [ ] Add ROCm detection and fallback; tune for gfx1032 (HSA_OVERRIDE).
  - [ ] Prototype enhancements: optical flow stub or FlowNSFW integration; VLM (local Qwen2.5-VL) for captions + keyword parse.
  - [ ] Gradio UI or CLI wrapper for batch processing.
- [ ] **Review:** Run on sample videos; verify tags, confidence logs, GPU usage (rocm-smi), no uploads; measure accuracy on known content.
- [ ] **Checkpoint 2:** Commit and push completed work.
- [ ] **Archive approval:** Ask user before archiving.
- [ ] **Checkpoint 3:** Archive and final push.

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ROCm installation/PyTorch compatibility on RX 6600 | Medium | High | Use HSA_OVERRIDE_GFX_VERSION=gfx1030; test with official ROCm PyTorch wheels; fallback to CPU path documented. |
| Model accuracy / false positives on "specific acts" | High | Medium | Use broad NSFW first; custom CLIP prompts for acts; VLM chaining as optional; user validates outputs. |
| Long videos / many frames performance | Medium | Medium | 1 fps sampling; batch inference; temp dir cleanup; GPU acceleration. |
| Bias in open NSFW models | Medium | Medium | Document limitation; recommend user testing on own data. |
| Minor-related content risk | Low | Critical | Hard rule: decline any clear request involving minors; pipeline never processes such content. |
| EU AI Act / legal for moderation use | Low | High | User responsibility; pipeline is personal/offline only. |

## Verification commands

```bash
# ROCm / GPU check
rocm-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"

# Basic model load test (Falconsai)
python -c "
from transformers import pipeline
classifier = pipeline('image-classification', model='Falconsai/nsfw_image_detection')
print(classifier('test.jpg'))
"

# Frame extraction test
ffmpeg -i sample.mp4 -vf "fps=1" -t 10 frames/frame%04d.png

# Full pipeline run (once implemented)
python video_nsfw_tagger.py --input sample.mp4 --output tags.json --threshold 0.7
```

## Progress log

- **09-07-2026 23:55** — To-do created from user-provided practical approach. Specialist input not yet dispatched for this specific pipeline (webhook specialists were for prior task).

## Notes

- All processing local; respect "decline if clear minor-related" rule.
- Models: Falconsai/nsfw_image_detection as primary; CLIP for flexible prompts; Qwen2.5-VL for VLM chaining.
- Tagging examples: JSON sidecar with {video_path, nsfw_percent, flagged_frames: [{frame, label, conf}], tags: ["nsfw"]}.
- References: FlowNSFW repo, LearnOpenCV CLIP+Gemini tutorial, HF model cards, ffmpeg docs.
- Writing style: Follow /home/elvee/agent-output/writer/standards/writing-style-guide.md (warm, clear, direct, practical).

## Next step

Confirm with Elvee: target categories (broad NSFW vs. specific acts), sample video formats/lengths, preferred output (JSON sidecar vs. DB), and whether to dispatch specialists for ROCm setup + model benchmarking before implementation.
