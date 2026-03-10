"""
AI worker script using Stable Video Diffusion (SVD) to generate motion clips.
Uses SD-Turbo for fast seed image generation.
"""

import sys
import os
import time
import json
import traceback
import torch
import random
from PIL import Image
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline

# Configuration
CLIP_FRAMES = 14  # SVD-XT supports 25, SVD supports 14
MOTION_BUCKET_ID = 127
NOISE_AUG_STRENGTH = 0.1
FPS = 7

def read_features(features_file):
    """Read JSON audio features written by main.py audio_thread."""
    try:
        with open(features_file, "r") as f:
            return json.loads(f.read())
    except Exception:
        return {}

def run(watch_dir, features_file):
    os.makedirs(watch_dir, exist_ok=True)
    
    fx_state_file = os.path.join(os.path.dirname(features_file), "vj_fx_state.json")
    if not os.path.exists(fx_state_file):
        with open(fx_state_file, "w") as f:
            f.write("{}")

    try:
        print("[AI VIDEO WORKER] Initializing SD-Turbo (Stage 1/2)...")
        # Load SD-Turbo for seed images
        sd_turbo = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        
        print("[AI VIDEO WORKER] Initializing Stable Video Diffusion (Stage 2/2)...")
        # Load SVD
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        
        print("[AI VIDEO WORKER] Applying VRAM Optimizations for 8GB GPU...")
        print("[AI WORKER] All models loaded and ready.")
        
        # 4070 8GB VRAM Optimizations
        print("[AI WORKER] Applying 8GB VRAM Optimizations...")
        sd_turbo.enable_model_cpu_offload()
        svd_pipe.enable_model_cpu_offload()
        svd_pipe.enable_vae_slicing()
        svd_pipe.enable_vae_tiling()
        
        # Ensure we use xformers if available or torch 2.0+ scaled dot product
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("[AI WORKER] Using native Scaled Dot Product Attention.")
        
        print("[AI WORKER] Memory optimization complete.")
    except Exception as e:
        print(f"[AI WORKER] Init failed: {e}")
        traceback.print_exc()
        open(os.path.join(watch_dir, "FAIL"), "w").close()
        return

    frame_idx = 0
    current_prompt = "a cosmic nebula with swirling colors of purple and blue, photorealistic"
    current_motion = 0.5
    
    while True:
        features = read_features(features_file)
        new_prompt = features.get("prompt", current_prompt)
        motion_intensity = features.get("motion_intensity", 0.5)
        
        # Audio energy reactivity
        rms = features.get("smoothed_rms", 0.0)
        bass = features.get("bass", 0.0)
        
        # Map agent motion_intensity (0-1) to motion_bucket_id (0-255)
        # Add a bit of audio jitter to the motion
        audio_motion_boost = min(bass / 5000.0, 0.2) if bass > 1000 else 0
        effective_motion = min(motion_intensity + audio_motion_boost, 1.0)
        bucket_id = int(effective_motion * 255)
        
        # Map RMS to noise augmentation (0.01 to 0.5)
        noise_aug = min(0.01 + (rms / 8000.0) * 0.5, 0.5)
        
        if new_prompt != current_prompt:
            current_prompt = new_prompt
            print(f"[AI] Generating new clip for: {current_prompt[:60]}...")
            print(f"[AI] Motion Intensity: {effective_motion:.2f} (Bucket: {bucket_id}), Noise Aug: {noise_aug:.3f}")

        try:
            # 1. Generate seed image
            gen_start = time.time()
            seed_image = sd_turbo(
                prompt=current_prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=512, height=288,   # 16:9, multiples of 64
            ).images[0]

            # SVD expects 512x288 (16:9) — no resize needed
            
            # 2. Generate Video Clip
            print(f"[AI] Generating {CLIP_FRAMES} frames with motion_bucket_id={bucket_id}...")
            frames = svd_pipe(
                seed_image, 
                decode_chunk_size=8, 
                generator=torch.manual_seed(random.randint(0, 1000000)),
                motion_bucket_id=bucket_id,
                noise_aug_strength=noise_aug,
                num_frames=CLIP_FRAMES
            ).frames[0]
            
            gen_time = time.time() - gen_start
            print(f"[AI] Clip generated in {gen_time:.2f}s")

            # 3. Save frames
            for i, frame in enumerate(frames):
                out_path = os.path.join(watch_dir, f"frame_{frame_idx:06d}.png")
                # Resize to target engine resolution if necessary
                frame.save(out_path, format="PNG")
                frame_idx += 1
                
            print(f"[AI] Saved frames up to {frame_idx}")

        except torch.cuda.OutOfMemoryError:
            print("[AI WORKER] CUDA Out of Memory! Clearing cache...")
            torch.cuda.empty_cache()
            time.sleep(5)
        except Exception as e:
            print(f"[AI WORKER] Generation failed: {e}")
            traceback.print_exc()
            time.sleep(2.0)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ai_worker_video.py <watch_dir> <features_file>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
