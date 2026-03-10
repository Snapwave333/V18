# VIBES — System Architecture

## Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AUDIO LAYER                                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ AudioIngest (PyAudio + FFT)                                           │  │
│  │  sub_bass [0-60Hz]  bass [60-250Hz]  mid [250-4kHz]  high [4-20kHz]  │  │
│  │  beat_detected · BPM · transient · energy_level · energy_trend        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│         │ audio_features dict (shared memory)                               │
└─────────┼───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  NARRATIVE + PROMPT LAYER                                                   │
│                                                                             │
│  ┌────────────────────────────────────┐   ┌───────────────────────────────┐ │
│  │ Storyteller                        │   │ Susa (Prompt Generator)       │ │
│  │  Dan Harmon 8-beat story circle    │   │  _UsageMemory — recency decay │ │
│  │  45s per beat                      │──►│  _ThemeMemory — novelty bias  │ │
│  │  Ollama LLM visual descriptions   │   │  PerformanceMemory — ML weights│ │
│  │  Few-shot exemplar learning        │   │  → subjects × styles × desc   │ │
│  │  ↑ records high-scoring beats      │   │  → ascii_force always 1.0     │ │
│  └────────────────────────────────────┘   └───────────────┬───────────────┘ │
│         ▲ record_score(beat, score)                       │ prompt string   │
└─────────┼─────────────────────────────────────────────────┼─────────────────┘
          │                                                  │
          │                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GENERATION LAYER  (subprocess)                                             │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ ai_worker_deform.py  [DEFAULT]  —  SD 1.5 + LCM-LoRA img2img        │   │
│  │                                                                      │   │
│  │   prev_frame ──► affine_warp(bass/mid/high) ──► img2img ──► frame   │   │
│  │                   zoom=bass   rot=mid   tx=high   strength~0.45      │   │
│  │                   4 steps · guidance_scale=1.0 · 640×360             │   │
│  │                   torch.compile() UNet · xformers attention          │   │
│  │                                                                      │   │
│  │  Every 5th frame: AestheticScorer.submit(frame) ──►                 │   │
│  │    ┌──────────────────────────────────────────────────────────────┐  │   │
│  │    │ AestheticScorer (CLIP ViT-B/32, async background thread)    │  │   │
│  │    │  score = sim(frame, POSITIVE) - sim(frame, NEGATIVE) → [0,1]│  │   │
│  │    │  callback → Susa.record_aesthetic(tokens, score)             │  │   │
│  │    │           → Storyteller.record_score(beat_name, score)      │  │   │
│  │    └──────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Alternates: ai_worker_optimized.py (SD-Turbo txt2img, faster/less coherent)│
│              ai_worker_video.py     (SVD video clips, highest quality)      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ frame PNG (atomic file swap)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FRAME CACHE BUFFER                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ deque(maxlen=960)  —  2 minutes @ 8fps                               │   │
│  │ Fills silently. Playback begins only when buffer is full.            │   │
│  │ Cohesive animation: watching 2 minutes ago, always a full scene.     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ PIL Image @ 8fps
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RENDER LAYER  (ModernGL GLSL @ 1920×1080)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ ShaderRenderer                                                       │   │
│  │                                                                      │   │
│  │  GPU Texture (640×360 upsampled)                                     │   │
│  │       │                                                              │   │
│  │       ▼                                                              │   │
│  │  GLSL Fragment Shader                                                │   │
│  │   ① ASCII atlas lookup — 16-density glyph chars, GPU texture        │   │
│  │       cell_size = 8px base + bass_breathing (8→13px)                │   │
│  │       per-cell color sampled from source frame                      │   │
│  │       char brightness = luminance of cell region                    │   │
│  │   ② Saturation boost × 1.4 (HSV) on each cell                      │   │
│  │   ③ Bright glyph render: cell_color × char_val × 2.2               │   │
│  │       + inter-character ambient: cell_color × (1-char_val) × 0.12  │   │
│  │   ④ [B] Bloom  — 8-neighbour cell sampling, weighted average       │   │
│  │   ⑤ [C] Chromatic aberration — RGB channel split on beats          │   │
│  │   ⑥ [L] Scanlines — sin-wave row modulation                        │   │
│  │   ⑦ Vignette — edge darkening via smoothstep                       │   │
│  │                                                                      │   │
│  │  Audio uniforms: u_bass, u_mid, u_high, u_beat, u_bpm              │   │
│  │  FX uniforms:    u_fx_bloom, u_fx_chroma, u_fx_scanlines           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │ pygame window  "Vibes VJ"  1920×1080                              │
└─────────┼───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│  OBS Window Capture             │
│  → Output: 3840×2160 (4K)       │
│  → Lanczos upscale              │
└─────────────────────────────────┘
```

---

## ML Learning Loop

```
Session N                               Session N+1
─────────────────────────────────────   ──────────────────────────────
Generate frame with tokens              Load data/performance_memory.json
      │                                        │
      ▼                                        ▼
AestheticScorer scores frame [0,1]      Susa._ml_weight(token)
      │                                   returns EMA score → [0.15, 2.0]
      ▼                                        │
PerformanceMemory.record(tokens, score)        ▼
  EMA α=0.3 per token                   token selection weighted by quality
  save every 40 calls                   → engine favors what looked good
      │
      ▼
Storyteller.record_score(beat, score)
  if score > 0.62:
    store context as beat exemplar
    (up to 8 per beat)
      │
      ▼ (next Ollama call)
  few-shot examples injected into prompt
  → LLM generates better descriptions
```

---

## Component Interactions

```
main.py
  │
  ├── AudioIngest ──────────────────────────────────────────────┐
  │       read: audio_features                                   │
  │                                                             │
  ├── Storyteller.set_audio_energy(level, trend)                 │
  │       feeds: audio energy level to narrative pacing          │
  │                                                             │
  ├── ai_worker_deform (subprocess via mp.Process)              │
  │       receives: prompt, audio_features via Queue             │
  │       sends:    frame PNG via atomic file write              │
  │       internal: AestheticScorer → callbacks to Susa/Story    │
  │                                                             │
  ├── Frame Cache Buffer (deque 960 frames)                      │
  │       polls: frame file every ~125ms                         │
  │       plays: 2 minutes behind live generation                │
  │                                                             │
  └── ShaderRenderer ◄──────────────────────────────────────────┘
          renders: buffered frames + audio uniforms in realtime
          hotkeys: B/C/L/F/Q
```

---

## File Structure

```
ollama-vj-engine/
├── README.md
├── ARCHITECTURE.md          ← this file
├── CHANGELOG.md
└── python/
    ├── main.py                  # Orchestrator + 2-min frame buffer
    ├── ai_worker_deform.py      # ★ img2img feedback loop (DEFAULT)
    ├── ai_worker_optimized.py   # SD-Turbo txt2img
    ├── ai_worker_video.py       # SVD video clips
    ├── audio_ingest.py          # FFT + beat detection
    ├── shader_renderer.py       # ModernGL GLSL + hotkeys
    ├── susa.py                  # AI prompt generator + ML weights
    ├── storyteller.py           # Dan Harmon narrative + few-shot
    ├── aesthetic_scorer.py      # CLIP async aesthetic scorer
    ├── performance_memory.py    # Persistent EMA quality memory
    ├── agent_bridge.py          # AI agent state bridge
    ├── loading_screen.py        # 2-min cinematic loading screen
    ├── requirements.txt
    └── data/
        ├── performance_memory.json   # auto-created, survives sessions
        └── storyteller_memory.json  # auto-created, survives sessions
```

---

## Performance Targets

| Metric           | Target          | Notes                              |
|------------------|-----------------|------------------------------------|
| Generation FPS   | 4–8 fps         | LCM-LoRA 4-step img2img            |
| Render FPS       | 60 fps          | ModernGL GLSL, GPU-bound           |
| Buffer delay     | 120s (default)  | 960 frames @ 8fps                  |
| Resolution (gen) | 640×360         | 16:9, SD 1.5 native                |
| Resolution (out) | 1920×1080       | Pygame window for OBS capture      |
| OBS output       | 3840×2160       | Lanczos upscale from 1080p         |
| VRAM usage       | ~4 GB           | SD 1.5 + LCM-LoRA + CLIP          |
| Compile time     | ~60s first run  | torch.compile() UNet cache         |
