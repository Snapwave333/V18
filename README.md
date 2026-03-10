<div align="center">
  <img src="python/favicon.png" alt="Vibes V18 Logo" width="200" />
  <h1>VIBES ⚡ V18</h1>
  <p><b>Live AI-Choreographed ASCII Diffusion Engine</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
  
  <p>
    <em>Real-time audio-reactive video diffusion rendered as pixel-perfect colored ASCII glyphs,<br>
    displayed at 1080p and captured by OBS for 4K output. Gets smarter every session.</em>
  </p>
</div>

---

## 🎵 What It Does

```mermaid
graph TD
    Mic[Microphone / System Audio] --> AI["AudioIngest<br>FFT, beat detect, BPM, spectral bands"]
    AI -->|Audio Data| Susa["Susa<br>AI prompt generator"]
    Susa -->|Themes & Subjects| Story["Storyteller<br>Dan Harmon narrative arc + Ollama"]
    AI -->|Beat / Frequencies| Worker["ai_worker_deform<br>SD 1.5 + LCM-LoRA img2img feedback loop<br>prev_frame → affine_warp → img2img → new_frame"]
    Story -->|Narrative Prompt| Worker
    Worker --> Cache["2-Minute Frame Cache Buffer<br>plays back with 2-min cohesive delay"]
    Cache --> Renderer["ShaderRenderer — ModernGL/GLSL @ 1920×1080<br>• Pixel-perfect ASCII glyph mapping<br>• Bass-reactive cell size breathing<br>• Visual FX: Bloom, Chromatic Aberration, Scanlines"]
    Renderer --> OBS["OBS Window Capture → 4K upscale output"]
```

---

## 🧠 ML Learning Loop

Every session the system gets smarter:

1. 🎯 **AestheticScorer** — CLIP ViT-B/32 rates each generated frame `[0→1]` asynchronously
2. 💾 **PerformanceMemory** — EMA score tracked per prompt element, saved to `data/performance_memory.json`
3. 🎲 **Susa** — multiplies ML quality weights into token selection probability → high-scoring subjects/styles chosen more
4. 📖 **Storyteller** — high-scoring visual contexts saved per story beat → primes Ollama as few-shot examples next session

*After a few hours of use, the engine self-tunes toward your visual aesthetic.*

---

## 🚀 Quick Start

### 1. Install Dependencies

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

### 2. Run the Engine

```bash
cd ollama-vj-engine/python
python main.py
```

> **Note:** The first run takes ~60s for `torch.compile()` to compile the UNet. Subsequent runs are incredibly fast.

#### Worker Modes:
```bash
python main.py --worker deform     # default: img2img feedback (smooth video)
python main.py --worker optimized  # SD-Turbo txt2img (faster, less coherent)
python main.py --worker video      # SVD video clips (highest quality, slowest)
python main.py --delay 60          # shorter cache delay (default 120s)
```

---

## 🎥 OBS Setup (4K Output)

1. Add **Window Capture** → select `Vibes VJ`
2. Set OBS **Output Resolution**: `3840×2160`
3. Set OBS **Canvas**: `1920×1080` *(let OBS handle the upscale)*
4. Set Source filter: **Lanczos** *(for the sharpest ASCII upscale)*

---

## ⌨️ Hotkeys
*Ensure the Vibes window is focused*

| Key | Effect |
|:---:|:---|
| <kbd>B</kbd> | Toggle **Bloom** glow |
| <kbd>C</kbd> | Toggle **Chromatic aberration** |
| <kbd>L</kbd> | Toggle **Scanlines** |
| <kbd>H</kbd> | Toggle **HUD Overlay** |
| <kbd>F</kbd> | Toggle **Fullscreen** |
| <kbd>Q</kbd> / <kbd>ESC</kbd> | Quit Application |

---

## 🏗️ Architecture

<details>
<summary><b>Click to View Full Architecture Diagram</b></summary>

```mermaid
flowchart TD
    subgraph Main ["main.py (Orchestrator)"]
        direction TB
        
        Audio["AudioIngest<br>sub_bass, bass, mid, high, centroid, beat / BPM"]
        
        subgraph Worker ["ai_worker_deform (Subprocess)"]
            direction TB
            W1["SD 1.5 + LCM-LoRA img2img<br>640×360 (16:9)"]
            W2["Deforum-style affine warp"]
            W3["torch.compile() UNet"]
            W4["AestheticScorer (CLIP async)"]
        end
        
        subgraph Susa ["Susa (Prompt AI)"]
            direction TB
            S1["_UsageMemory — time-decay recency"]
            S2["_ThemeMemory — thematic drift prevention"]
            S3["PerformanceMemory — CLIP score learning"]
        end
        
        subgraph Storyteller ["Storyteller (Dan Harmon Narrative)"]
            direction TB
            D1["45s per beat, Ollama LLM descriptions"]
            D2["Few-shot exemplar learning (disk persist)"]
        end
        
        Buffer["2-Minute Frame Cache Buffer<br>deque, 960 frames @ 8fps"]
        
        Renderer["ShaderRenderer (ModernGL + pygame, 1920×1080)<br>• GPU ASCII Atlas Mapping<br>• FX: Bloom, Chromatic Aberration, Scanlines, Vignette<br>• Audio-reactive shader uniforms"]
        
        Spout["Optional Spout/Syphon Output<br>Resolume / MadMapper integration"]

        Audio -- "audio_features" --> Susa
        Susa -- "prompt" --> Storyteller
        Storyteller -- "narrative prompt" --> Worker
        Audio -- "audio/beat reactive warp" --> Worker
        W4 -- "ML score callback" --> S3
        
        Worker -- "rendered frames" --> Buffer
        Buffer --> Renderer
        Renderer --> Spout
    end
    
    Disk[("(Persistent Disk)")]
    S3 -. "saves to" .-> Disk
    D2 -. "saves to" .-> Disk
```
</details>

---

## 📁 File Structure

```text
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
├── loading_screen.py        # Cinematic 2-min loading screen
└── hud.py                   # On-screen overlay and Spout toggle
data/                        # (Auto-created on first run)
├── performance_memory.json  # Susa's learned quality weights
└── storyteller_memory.json  # Beat exemplars for Ollama
```

---
<div align="center">
  <sub>Built with ⚡ for Live VJ Performances</sub>
</div>
