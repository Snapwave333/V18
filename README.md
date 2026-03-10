# VIBES — Live AI-Choreographed ASCII Diffusion Engine

Real-time audio-reactive video diffusion rendered as pixel-perfect colored ASCII glyphs,
displayed at 1080p and captured by OBS for 4K output. Gets smarter every session.

---

## What It Does

```
[Microphone / System Audio]
        │
        ▼
[AudioIngest — FFT, beat detect, BPM, spectral bands]
        │
        ├──► [Susa — AI prompt generator (learns what looks good over time)]
        │           │
        │           ▼
        │    [Storyteller — Dan Harmon narrative arc + Ollama few-shot learning]
        │
        ▼
[ai_worker_deform — SD 1.5 + LCM-LoRA img2img feedback loop]
   prev_frame → affine_warp(bass/mid/high) → img2img → new_frame
        │
        ▼
[2-Minute Frame Cache Buffer — plays back with 2-min cohesive delay]
        │
        ▼
[ShaderRenderer — ModernGL/GLSL @ 1920×1080]
   ● Pixel-perfect ASCII glyph mapping (GPU, ~270×152 chars dense)
   ● Bass-reactive cell size breathing
   ● Bloom glow  [B]  · Chromatic aberration on beats  [C]  · Scanlines  [L]
   ● Hue shift, saturation boost, edge glow, FX modes (mirror/quad/kaleido)
        │
        ▼
[OBS Window Capture → 4K upscale output]
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  main.py  (orchestrator)                                            │
│  ┌──────────────┐  ┌──────────────────────────────────────────────┐ │
│  │ AudioIngest  │  │ ai_worker_deform  (subprocess)               │ │
│  │  sub_bass    │  │  SD 1.5 + LCM-LoRA img2img                  │ │
│  │  bass        │  │  640×360 (16:9)                              │ │
│  │  mid         │  │  Deforum-style affine warp                   │ │
│  │  high        │  │  torch.compile() UNet                        │ │
│  │  centroid    │  │  AestheticScorer (CLIP async)                │ │
│  │  beat / BPM  │  │       │ score feedback                       │ │
│  └──────┬───────┘  └───────┼──────────────────────────────────────┘ │
│         │ audio_features   │ ML score callback                       │
│         ▼                  ▼                                         │
│  ┌────────────────────────────────────────────┐                      │
│  │ Susa  (prompt AI)                          │                      │
│  │  _UsageMemory  — time-decay recency        │                      │
│  │  _ThemeMemory  — thematic drift prevention │                      │
│  │  PerformanceMemory — CLIP score learning   │ ← persists to disk   │
│  └────────────────────────────────────────────┘                      │
│         │ prompt                                                      │
│         ▼                                                            │
│  ┌────────────────────────────────────────────┐                      │
│  │ Storyteller  (Dan Harmon 8-beat narrative) │                      │
│  │  45s per beat, Ollama LLM descriptions     │                      │
│  │  Few-shot exemplar learning (disk persist) │ ← learns over time   │
│  └────────────────────────────────────────────┘                      │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ 2-Minute Frame Cache Buffer (deque, 960 frames @ 8fps)           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ ShaderRenderer  (ModernGL + pygame, 1920×1080)                   │ │
│  │  GLSL Fragment Shader:                                           │ │
│  │   • ASCII atlas — GPU texture lookup, 16-density levels         │ │
│  │   • Neighbour-cell bloom glow           [B to toggle]           │ │
│  │   • Beat-reactive chromatic aberration  [C to toggle]           │ │
│  │   • CRT scanlines                       [L to toggle]           │ │
│  │   • Vignette, crossfade, audio uniforms                        │ │
│  │   • FX: mirror, quad-split, kaleidoscope                       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ML Learning Loop

Every session the system gets smarter:

1. **AestheticScorer** — CLIP ViT-B/32 rates each generated frame [0→1] asynchronously
2. **PerformanceMemory** — EMA score tracked per prompt element, saved to `data/performance_memory.json`
3. **Susa** — multiplies ML quality weights into token selection probability → high-scoring subjects/styles chosen more
4. **Storyteller** — high-scoring visual contexts saved per story beat → primes Ollama as few-shot examples next session

After a few hours of use, the engine self-tunes toward your visual aesthetic.

---

## Install

```bash
cd ollama-vj-engine/python

# Fix torch + install deps
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.37.0 transformers accelerate safetensors \
  xformers opencv-python moderngl pygame pyaudio

# Optional: Ollama for narrative generation
# Install from https://ollama.ai then: ollama pull llama3.2
```

---

## Run

```bash
cd ollama-vj-engine/python
python main.py
```

First run takes ~60s for torch.compile() to compile the UNet. Subsequent runs are fast (cached).

**Worker flags:**
```bash
python main.py --worker deform     # default: img2img feedback (smooth video)
python main.py --worker optimized  # SD-Turbo txt2img (faster, less coherent)
python main.py --worker video      # SVD video clips (highest quality, slowest)
python main.py --delay 60          # shorter cache delay (default 120s)
```

---

## OBS Setup (4K Output)

1. Add **Window Capture** → select `Vibes VJ`
2. OBS **Output Resolution**: `3840×2160`
3. OBS **Canvas**: `1920×1080` (let OBS upscale)
4. Source filter: **Lanczos** for sharpest ASCII upscale

---

## Hotkeys (focus the Vibes window)

| Key | Effect |
|-----|--------|
| `B` | Toggle **Bloom** glow |
| `C` | Toggle **Chromatic aberration** |
| `L` | Toggle **Scanlines** |
| `F` | Toggle fullscreen |
| `Q` / `ESC` | Quit |

---

## File Structure

```
python/
├── main.py                  # Orchestrator + 2-min frame buffer
├── ai_worker_deform.py      # ★ img2img feedback loop (DEFAULT)
├── ai_worker_optimized.py   # SD-Turbo txt2img
├── ai_worker_video.py       # SVD video clips
├── audio_ingest.py          # Full spectral audio analysis
├── shader_renderer.py       # ModernGL GLSL renderer + hotkeys
├── susa.py                  # AI prompt generator with ML weights
├── storyteller.py           # Narrative arc + few-shot learning
├── aesthetic_scorer.py      # CLIP async frame scorer
├── performance_memory.py    # Persistent ML quality memory
├── agent_bridge.py          # AI agent state bridge
├── loading_screen.py        # Cinematic 2-min loading screen
└── data/
    ├── performance_memory.json   # Susa's learned quality weights (auto-created)
    └── storyteller_memory.json  # Beat exemplars for Ollama (auto-created)
```
