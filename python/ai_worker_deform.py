"""
Deform worker — img2img feedback loop with Deforum-style 2D affine warping.

Fixes the slideshow/PowerPoint problem by deriving each frame from the previous:

    prev_frame → affine_warp(audio) → img2img(strength=0.4-0.65) → new_frame
                                              ↑
                                    audio params modulate this

This creates temporal coherence: each frame inherits structure from the last,
so output looks like flowing video not a slideshow.

Model: SD 1.5 + LCM-LoRA img2img
  - 4 denoising steps via LCM (Latent Consistency Model)
  - No CFG (guidance_scale=1.0 for LCM)
  - ~4-8 FPS at 640x360 on RTX 4070 Laptop

Resolution: 640x360 (16:9 native)
"""

import sys
import os
import time
import json
import traceback

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler, AutoencoderTiny
from PIL import Image
from shared_memory_ipc import SharedFrameMemory
# ── Config ────────────────────────────────────────────────────────────────────
GEN_W = 448
GEN_H = 256  # 16:9, slightly lower res for 2x faster generation
NUM_STEPS = 4       # LCM: 4 steps = good quality/speed balance
GUIDANCE  = 2.0     # CFG 2.0: makes negative prompt actually suppress abstracts

# img2img strength: LOW = model starts from low noise = refines detail = realistic output
# HIGH strength pushes LCM into near-txt2img mode which hallucinates abstract blobs
STRENGTH_BASE  = 0.50   # sweet spot: preserves structure + refines detail each frame
STRENGTH_RANGE = 0.10   # audio-driven range above base

# Force a fresh txt2img reseed every N frames to prevent content from accumulating/going abstract
RESEED_INTERVAL = 80    # ~20s at 4fps — keeps scenes refreshing without flickering

# Silence gating — skip generation when audio is below this RMS threshold.
# Holds the last frame still instead of burning GPU on silence.
SILENCE_RMS_THRESHOLD = 80    # below this = silent (tune if mic noise floor is higher)
SILENCE_SLEEP        = 0.10   # seconds to sleep per idle loop (saves GPU/CPU)


# Affine warp amplitude multipliers (CRITICALLY LOWERED to stop "abstract blur")
ZOOM_DEFAULT     = 1.002   # Constant slow zoom in for "depth" feel
ROT_AUDIO_SCALE  = 0.04     # mid → rotation (degrees) - keep it subtle
TX_AUDIO_SCALE   = 0.8      # high → x-translation (pixels)
TY_AUDIO_SCALE   = 0.6      # bass → y-translation (pixels)

PROMPT_INTERVAL = 4   # re-evaluate prompt more frequently to break abstract loops

MODEL_ID    = "runwayml/stable-diffusion-v1-5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

# Import the full anti-pattern negative prompt from susa — it blocks fractals,
# kaleidoscopes, tessellations, abstract noise etc. that build up in feedback loops.
# The old local version was missing ALL of these, causing the repeating-pattern problem.
from susa import NEGATIVE_PROMPT  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_features(features_file: str) -> dict:
    try:
        with open(features_file, "r") as f:
            return json.loads(f.read())
    except Exception:
        return {}


def affine_warp(frame_np: np.ndarray, zoom: float, angle: float,
                tx: float, ty: float) -> np.ndarray:
    """
    Deforum-style 2D affine warp.
    zoom  : scale factor (1.0 = no zoom, 1.002 = slow zoom in)
    angle : degrees of rotation
    tx/ty : translation in pixels
    Uses BORDER_REFLECT_101 to avoid black edges at frame boundary.
    """
    h, w = frame_np.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, zoom)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(
        frame_np, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def audio_to_warp_params(features: dict) -> dict:
    """Map audio features → warp + diffusion strength params."""
    bass = float(features.get("bass", 0.0))
    mid  = float(features.get("mid",  0.0))
    high = float(features.get("high", 0.0))

    # Soft normalization — tuned to typical RMS/band magnitudes
    bass_n = min(bass / 3000.0, 1.0)
    mid_n  = min(mid  / 2000.0, 1.0)
    high_n = min(high / 1500.0, 1.0)

    # Gentle zoom in by default, briefly scale up on high-frequency energy
    zoom = ZOOM_DEFAULT + (high_n * 0.005)

    # Gentle rotation from mid content (centred around 0)
    rotation = (mid_n - 0.5) * ROT_AUDIO_SCALE

    # Shimmer/translate from high-freq energy
    tx = (high_n - 0.5) * TX_AUDIO_SCALE
    ty = (bass_n - 0.5) * TY_AUDIO_SCALE

    # Fixed strength avoids CUDA kernel recompilation on varying step count
    # With LCM 4 steps, strength=0.75 → 3 steps consistently
    strength = STRENGTH_BASE

    return {"zoom": zoom, "rotation": rotation, "tx": tx, "ty": ty,
            "strength": strength}


# ── Main worker ───────────────────────────────────────────────────────────────

def run(watch_dir: str, features_file: str):
    os.makedirs(watch_dir, exist_ok=True)

    fx_state_file = os.path.join(os.path.dirname(features_file), "vj_fx_state.json")
    with open(fx_state_file, "w") as f:
        f.write("{}")

    # ── Load model ──────────────────────────────────────────────────────────
    try:
        print(f"[DEFORM] Loading {MODEL_ID} + LCM-LoRA img2img pipeline...")
        pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")

        print("[DEFORM] Loading TAESD (Tiny VAE)...")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to("cuda")

        print(f"[DEFORM] Loading LCM-LoRA: {LCM_LORA_ID}")
        pipe.load_lora_weights(LCM_LORA_ID)
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        # xformers provides ~30% speedup
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[DEFORM] xformers memory efficient attention enabled.")
        except Exception:
            print("[DEFORM] Using default attention.")

        # Enable cuDNN benchmarking for fixed resolution performance boost
        torch.backends.cudnn.benchmark = True

        # xformers provides ~30% speedup already; skip torch.compile to avoid
        # multi-minute CUDA kernel compile on first run (Windows compatibility issue)

        print(f"[DEFORM] Ready — {GEN_W}x{GEN_H} (16:9) img2img feedback loop.")

    except Exception as e:
        print(f"[DEFORM] Init failed: {e}")
        traceback.print_exc()
        open(os.path.join(watch_dir, "FAIL"), "w").close()
        return

    # ── Initialize Shared Memory ────────────────────────────────────────────
    try:
        shm_out = SharedFrameMemory("vj_frame_buffer", GEN_W, GEN_H, create=True)
    except Exception as e:
        print(f"[DEFORM] Warning: Failed to create SHM: {e}")
        shm_out = None

    # ── Initialize Spout sender (Windows → Resolume) ─────────────────────────
    spout_sender = None
    try:
        from SpoutGL import SpoutSender
        spout_sender = SpoutSender()
        spout_sender.init("VibesV18", GEN_W, GEN_H)
        print(f"[DEFORM] Spout sender 'VibesV18' ready — {GEN_W}x{GEN_H}")
    except Exception as e:
        print(f"[DEFORM] Spout not available ({e}) — running without Spout output.")

    # ── Seed frame ──────────────────────────────────────────────────────────
    # Generate a real scene via txt2img so the feedback loop starts from content,
    # not grey noise (grey → 2 denoising steps → abstract texture → loop lock-in).
    print("[DEFORM] Generating seed frame via txt2img...")
    try:
        from diffusers import AutoPipelineForText2Image
        seed_pipe = AutoPipelineForText2Image.from_pipe(pipe).to("cuda")
        with torch.inference_mode():
            seed_result = seed_pipe(
                prompt="a dramatic cinematic landscape, volumetric lighting, highly detailed, photorealistic",
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=4,
                guidance_scale=7.5,
                width=GEN_W,
                height=GEN_H,
            )
        current_frame = seed_result.images[0]
        del seed_pipe
        torch.cuda.empty_cache()
        print("[DEFORM] Seed frame generated.")
    except Exception as e:
        print(f"[DEFORM] Seed txt2img failed ({e}), using noise fallback.")
        seed_np = np.random.randint(30, 80, (GEN_H, GEN_W, 3), dtype=np.uint8)
        current_frame = Image.fromarray(seed_np)

    # ── Storytelling + prompt generation ────────────────────────────────────
    from susa import Susa
    from storyteller import get_storyteller
    from music_psychology import get_mapper as get_psych_mapper
    from color_theory import get_mapper as get_color_mapper

    susa         = Susa()
    storyteller  = get_storyteller()
    psych_mapper = get_psych_mapper()
    color_mapper = get_color_mapper()
    storyteller.start()

    # ── Aesthetic scoring → ML feedback loop ────────────────────────────────
    from aesthetic_scorer import AestheticScorer
    scorer = AestheticScorer()
    scorer.start()

    def _on_score(score: float, context: dict):
        """Callback: feed aesthetic score back into Susa's learning memory."""
        tokens = context.get("tokens", [])
        if tokens:
            susa.record_aesthetic(tokens, score)
        beat_name = context.get("beat_name", "")
        if beat_name:
            storyteller.record_score(beat_name, score)

    # Score every Nth frame to avoid scoring overhead (CLIP runs async so cost is low)
    SCORE_EVERY = 3

    frame_idx           = 0
    frames_since_prompt = PROMPT_INTERVAL   # force prompt on frame 0
    frames_since_seed   = RESEED_INTERVAL   # force reseed on frame 0 (already done above)
    
    # Immediately fetch the first story beat to avoid "placeholder" visuals
    initial_features = {"smoothed_rms": 0.1, "bass": 0.1}
    initial_narrative = storyteller.get_context()
    current_prompt = susa.generate_prompt(initial_features, initial_narrative)
    
    frame_times = []

    # Flush ML memory on any exit (KeyboardInterrupt, SIGTERM, etc.)
    import atexit
    from performance_memory import get_memory as _get_perf_memory
    atexit.register(lambda: _get_perf_memory().flush())
    atexit.register(lambda: storyteller.stop())

    # ── Pattern drift detection ───────────────────────────────────────────────
    # Counts consecutive low-scoring frames. When a run of abstract/repetitive
    # frames is detected, a reset is queued and handled in the main loop.
    _DRIFT_THRESHOLD  = 0.42   # aesthetic score below this = "going abstract"
    _DRIFT_MAX_FRAMES = 6      # reset after this many consecutive bad frames
    _drift = {"counter": 0, "reset_requested": False}

    def _on_score(score: float, context: dict):
        """Callback: feed aesthetic score back into Susa's learning memory.
        Also tracks drift; queues a seed-frame reset if visuals go abstract."""
        tokens = context.get("tokens", [])
        if tokens:
            susa.record_aesthetic(tokens, score)
        beat_name = context.get("beat_name", "")
        if beat_name:
            storyteller.record_score(beat_name, score)
        # Drift detection
        if score < _DRIFT_THRESHOLD:
            _drift["counter"] += 1
            if _drift["counter"] >= _DRIFT_MAX_FRAMES:
                _drift["counter"] = 0
                _drift["reset_requested"] = True
                print(f"[DEFORM] Pattern drift (score={score:.2f} x{_DRIFT_MAX_FRAMES}) — queuing seed reset.")
        else:
            _drift["counter"] = 0

    # ── Generation loop ──────────────────────────────────────────────────────
    while True:
        gen_start = time.time()

        # ── Periodic reseed + drift reset — both use txt2img for real content ──
        need_reseed = (
            _drift["reset_requested"]
            or frames_since_seed >= RESEED_INTERVAL
        )
        if need_reseed:
            if _drift["reset_requested"]:
                print("[DEFORM] Drift reset — regenerating seed frame via txt2img.")
            else:
                print(f"[DEFORM] Periodic reseed (every {RESEED_INTERVAL} frames) — injecting fresh scene.")
            _drift["reset_requested"] = False
            frames_since_seed = 0
            frames_since_prompt = PROMPT_INTERVAL  # force fresh prompt immediately
            try:
                # Softer reseed: strength=0.85 allows the new scene to 'emerge' 
                # from the previous structure, preventing a hard visual jump.
                with torch.inference_mode():
                    result = pipe(
                        prompt=current_prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=current_frame,
                        strength=0.85, 
                        num_inference_steps=6,
                        guidance_scale=7.5,
                        width=GEN_W,
                        height=GEN_H,
                    )
                current_frame = result.images[0]
            except Exception as e:
                print(f"[DEFORM] Reseed failed ({e}), using noise fallback.")
                current_frame = Image.fromarray(
                    np.random.randint(30, 80, (GEN_H, GEN_W, 3), dtype=np.uint8)
                )

        # ── Prompt update ────────────────────────────────────────────────────
        if frames_since_prompt >= PROMPT_INTERVAL:
            features  = read_features(features_file)

            # Map audio → psychology → color every prompt cycle
            psychology = psych_mapper.map(features)
            color      = color_mapper.map(psychology)

            # Push psychology/color to storyteller so Ollama scenes stay in mood
            storyteller.set_psychology(psychology, color)
            # Keep storyteller audio-energy in sync with psychology arousal
            # Derive energy trend from BPM delta (tempo accel/decel) + tension
            bpm_delta = psychology.get("bpm_delta", 0.0)
            if bpm_delta > 3.0:
                trend = "rising"
            elif bpm_delta < -3.0:
                trend = "falling"
            elif psychology["tension"] > 0.6:
                trend = "rising"
            elif psychology["tension"] < 0.3:
                trend = "falling"
            else:
                trend = "sustained"
            storyteller.set_audio_energy(
                "high" if psychology["arousal"] > 0.65 else
                ("low" if psychology["arousal"] < 0.35 else "medium"),
                trend,
            )

            narrative = storyteller.get_context()
            beat_name = storyteller.get_beat_name()
            new_prompt = susa.generate_prompt(features, narrative,
                                              psychology=psychology, color=color)

            if new_prompt != current_prompt:
                current_prompt = new_prompt
                rms  = features.get("smoothed_rms", 0)
                beat = features.get("beat", False)
                print(f"[DEFORM] rms={rms:.0f} beat={'Y' if beat else 'N'} "
                      f"mood={psychology['mood_label']} story='{beat_name}'")
                print(f"         prompt='{current_prompt[:80]}...'")

            # Write FX state for the shader renderer
            fx_state = susa.generate_fx_state(features, beat_name)
            try:
                tmp = fx_state_file + ".tmp"
                with open(tmp, "w") as f:
                    f.write(json.dumps(fx_state))
                os.replace(tmp, fx_state_file)
            except Exception:
                pass

            frames_since_prompt = 0
        else:
            features = read_features(features_file)

        # ── Silence gate — hold last frame, skip GPU work ────────────────────
        rms_now = float(features.get("smoothed_rms", 0.0))
        if rms_now < SILENCE_RMS_THRESHOLD:
            time.sleep(SILENCE_SLEEP)
            continue

        # ── Warp previous frame (temporal coherence) ────────────────────────
        warp = audio_to_warp_params(features)

        frame_np = np.array(current_frame)
        warped_np = affine_warp(
            frame_np,
            zoom=warp["zoom"],
            angle=warp["rotation"],
            tx=warp["tx"],
            ty=warp["ty"],
        )
        warped_pil = Image.fromarray(warped_np)

        # ── img2img denoising ────────────────────────────────────────────────
        try:
            with torch.inference_mode():
                result = pipe(
                    prompt=current_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=warped_pil,
                    strength=warp["strength"],
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE,
                    width=GEN_W,
                    height=GEN_H,
                )
            current_frame = result.images[0]

            # ── Check HUD settings for Spout toggle ─────────────────────────
            hud_spout_enabled = True
            try:
                hud_json = os.path.join(os.path.dirname(__file__), "data", "hud_settings.json")
                if os.path.exists(hud_json):
                    with open(hud_json, "r") as f:
                        hud_spout_enabled = json.load(f).get("spout", True)
            except Exception:
                pass

            # ── Send to Resolume via Spout ────────────────────────────────────
            if spout_sender and hud_spout_enabled:
                try:
                    frame_rgba = np.array(current_frame.convert("RGBA"), dtype=np.uint8)
                    spout_sender.sendImageData(frame_rgba.tobytes(), GEN_W, GEN_H)
                except Exception:
                    pass

            # ── Write to Shared Memory OR Disk as fallback ───────────────────
            if shm_out:
                try:
                    frame_np_out = np.array(current_frame)
                    shm_out.write_frame(frame_np_out, frame_idx)
                except Exception as e:
                    print(f"[DEFORM] SHM Write Error: {e}")
                # Always keep latest_frame.png updated for HTTP serving
                try:
                    tmp_latest = os.path.join(watch_dir, "latest_frame.tmp")
                    current_frame.save(tmp_latest, format="PNG")
                    os.replace(tmp_latest, os.path.join(watch_dir, "latest_frame.png"))
                except Exception:
                    pass
            else:
                tmp_path = os.path.join(watch_dir, f"frame_{frame_idx:06d}.tmp")
                out_path = os.path.join(watch_dir, f"frame_{frame_idx:06d}.png")
                current_frame.save(tmp_path, format="PNG")
                os.replace(tmp_path, out_path)

            # ── Aesthetic scoring → ML feedback (every Nth frame) ────────────
            if frame_idx % SCORE_EVERY == 0:
                scorer.submit(
                    current_frame,
                    _on_score,
                    context={
                        "tokens":    list(susa.last_tokens),
                        "beat_name": storyteller.get_beat_name(),
                    },
                )

            # Remove previous frame to save disk space if using disk
            if not shm_out and frame_idx > 0:
                prev = os.path.join(watch_dir, f"frame_{frame_idx - 1}.png")
                try:
                    os.remove(prev)
                except Exception:
                    pass

            frame_idx           += 1
            frames_since_prompt += 1
            frames_since_seed   += 1

            # ── Performance log ──────────────────────────────────────────────
            gen_time = time.time() - gen_start
            frame_times.append(gen_time)
            if len(frame_times) > 60:
                frame_times.pop(0)

            if frame_idx % 20 == 0:
                avg = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg if avg > 0 else 0
                print(
                    f"[DEFORM] frame={frame_idx:5d} "
                    f"avg={avg:.2f}s ({fps:.1f}fps) "
                    f"str={warp['strength']:.2f} "
                    f"zoom={warp['zoom']:.4f} "
                    f"rot={warp['rotation']:.3f}"
                )

        except torch.cuda.OutOfMemoryError:
            print("[DEFORM] CUDA OOM — clearing cache and shrinking frame.")
            torch.cuda.empty_cache()
            # Recover: reset to smaller noise seed
            current_frame = Image.fromarray(
                np.random.randint(32, 224, (GEN_H, GEN_W, 3), dtype=np.uint8)
            )
            time.sleep(2.0)

        except Exception as e:
            print(f"[DEFORM] Generation error: {e}")
            traceback.print_exc()
            time.sleep(1.0)
            
        # Synchronization micro-sleep:
        # Prevents PyTorch from hogging 100% of the CUDA command queue,
        # allowing PyGame's OpenGL renderer to grab the GPU to achieve 60 FPS
        torch.cuda.synchronize()
        time.sleep(0.005)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ai_worker_deform.py <watch_dir> <features_file>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
