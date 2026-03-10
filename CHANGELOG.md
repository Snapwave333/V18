# CHANGELOG

## v18 — Live AI-Choreographed ASCII Diffusion Engine

### Added

#### Core Engine
- **`ai_worker_deform.py`** — New default worker. SD 1.5 + LCM-LoRA img2img feedback loop
  - Deforum-style 2D affine warp driven by audio: zoom=bass, rotation=mid, tx/ty=high
  - 4-step img2img at guidance_scale=1.0 — temporal coherence, no more slideshow effect
  - 640×360 (16:9) output resolution
  - `torch.compile()` UNet — 20–35% speedup after first compile
  - xformers memory-efficient attention — VRAM savings
  - Atomic PNG frame saves (write to `.tmp` then `os.replace`)

#### Machine Learning Loop
- **`aesthetic_scorer.py`** — Async CLIP ViT-B/32 frame quality scorer
  - Scores every generated frame [0→1] against positive/negative aesthetic anchors
  - Non-blocking: runs in a background thread, never stalls generation
  - Feeds scores back into Susa and Storyteller for continuous learning

- **`performance_memory.py`** — Persistent EMA quality tracking
  - EMA (α=0.3) per prompt token, tracks aesthetic scores across sessions
  - Saves to `data/performance_memory.json` every 40 score recordings
  - `quality_weight(token)` returns [0.15, 2.0] multiplier for token selection

- **`susa.py`** — ML weight integration
  - Token selection probability now multiplied by `PerformanceMemory.quality_weight()`
  - `last_tokens` attribute exposes selected tokens to scoring callback
  - `record_aesthetic(tokens, score)` routes CLIP scores into PerformanceMemory
  - `ascii_force` always 1.0 — ASCII glyphs locked as focal point

- **`storyteller.py`** — Few-shot exemplar learning
  - High-scoring visual descriptions (score ≥ 0.62) saved per story beat
  - Up to 8 exemplars per beat, lowest-score evicted when at capacity
  - Exemplars injected as few-shot examples into Ollama prompts next session
  - Saves to `data/storyteller_memory.json`

#### Visual Effects (GPU, GLSL)
- **Bloom glow** `[B]` — 8-neighbour cell sampling with distance falloff
- **Chromatic aberration** `[C]` — RGB channel split, beat-reactive magnitude
- **CRT scanlines** `[L]` — sin-wave row modulation
- **Vignette** — always-on edge darkening via smoothstep
- **Saturation boost** — × 1.4 HSV boost per cell before glyph render
- **Brighter glyphs** — `cell_color × char_val × 2.2` (was implicit 1.0)
- **Inter-character ambient** — `cell_color × (1 - char_val) × 0.12` glow between chars

#### ASCII Rendering
- Base cell size reduced 14px → 8px — ~270 columns at 1920px (pixel-perfect density)
- Bass breathing range updated to 8→13px (was 10→18px)
- `ASCII_ON_FRACTION = 1.0` — ASCII always active, never fades

#### 2-Minute Frame Cache Buffer
- `deque(maxlen=960)` at 8fps = 120 seconds of buffered frames
- Playback waits until buffer is full — ensures cohesive animation, never a fragment

#### Loading Screen
- `loading_screen.py` — cinematic ASCII countdown while buffer fills

### Changed

- **Default worker**: `main.py` now starts `--worker deform` (was `optimized`)
- **`ai_worker_optimized.py`**: Output changed to 640×360 16:9 (was 512×512)
- **`ai_worker_video.py`**: Fixed `turbo_pipe` NameError → `sd_turbo`; seed image changed to 512×288 16:9
- **`requirements.txt`**: torch 2.2.2 → 2.4.1+cu121, diffusers 0.24.0 → 0.37.0, added opencv-python, updated xformers

### Fixed

- **Slideshow/PowerPoint problem** — txt2img generated unrelated frames each pass. Fixed by switching to img2img feedback loop; each frame is derived from the previous warped frame.
- **`turbo_pipe` NameError** in ai_worker_video.py — variable was named `sd_turbo` at load but `turbo_pipe` at inference call.
- **Broken torch install** (`~orch` invalid distribution) — uninstall and reinstall with correct CUDA index URL.
- **16:9 aspect ratio** — all workers now output 16:9 frames, shader renders at 1920×1080 16:9.

### Architecture

- Full pipeline documented in `ARCHITECTURE.md` (completely rewritten)
- README rewritten with architecture diagram, ML loop, install guide, hotkey table
- `data/` directory auto-created on first run; JSON memory files persist across sessions

---

## Hotkeys

| Key | Effect |
|-----|--------|
| `B` | Toggle Bloom glow |
| `C` | Toggle Chromatic aberration |
| `L` | Toggle CRT Scanlines |
| `F` | Toggle fullscreen |
| `Q` / `ESC` | Quit |

---

## Worker Modes

```bash
python main.py --worker deform     # default: img2img feedback (smooth video)
python main.py --worker optimized  # SD-Turbo txt2img (faster, less coherent)
python main.py --worker video      # SVD video clips (highest quality, slowest)
python main.py --delay 60          # shorter cache delay (default 120s)
```
