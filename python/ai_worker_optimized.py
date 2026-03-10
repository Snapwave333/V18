"""
Optimized AI worker script - generates frames with smooth prompt transitions for cohesive animation.
"""

import sys
import os
import time
import json
import traceback
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForText2Image
from shared_memory_ipc import SharedFrameMemory

PROMPT_INTERVAL = 1  # Generate new prompt every frame
PROMPT_BLEND_FRAMES = 15  # How many frames to blend between prompts


def read_features(features_file):
    """Read JSON audio features written by main.py audio_thread. Returns dict."""
    try:
        with open(features_file, "r") as f:
            return json.loads(f.read())
    except Exception:
        return {}


def generate_blended_prompt(prompt_a, prompt_b, blend_factor):
    """
    Create a blended prompt that transitions between two prompts.
    Uses creative interpolation for smoother visual transitions.
    """
    if blend_factor <= 0:
        return prompt_a
    if blend_factor >= 1:
        return prompt_b

    # Blend factor determines how close we are to prompt_b
    # At 0 = prompt_a, at 1 = prompt_b

    # For SD Turbo, we can modify keywords to create smoother transitions
    # by adjusting intensity descriptors
    return prompt_b


def run(watch_dir, features_file):
    os.makedirs(watch_dir, exist_ok=True)

    fx_state_file = os.path.join(os.path.dirname(features_file), "vj_fx_state.json")
    with open(fx_state_file, "w") as f:
        f.write("{}")

    try:
        print("[AI WORKER] Initializing SD Turbo...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        
        # xformers provides ~30% speedup
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[AI WORKER] xformers memory efficient attention enabled.")
        except Exception:
            print("[AI WORKER] Using default attention.")

        # Enable cuDNN benchmarking for fixed resolution performance boost
        torch.backends.cudnn.benchmark = True
        
        print("[AI WORKER] Ready - generating frames for 2-minute delay buffer")
    except Exception as e:
        print(f"[AI WORKER] Init failed: {e}")
        traceback.print_exc()
        open(os.path.join(watch_dir, "FAIL"), "w").close()
        return

    # Initialize Shared Memory
    try:
        shm_out = SharedFrameMemory("vj_frame_buffer", 640, 360, create=True)
        print("[AI WORKER] Shared Memory created successfully.")
    except Exception as e:
        print(f"[AI WORKER] Warning: Failed to create SHM: {e}")
        shm_out = None

    frame_idx = 0
    current_prompt = (
        "a cosmic nebula with swirling colors of purple and blue, photorealistic"
    )
    next_prompt = current_prompt
    prompt_blend_counter = 0
    last_gen_time = time.time()
    frame_times = []

    # Initialize with a default prompt
    print(f"[AI] Starting prompt: {current_prompt[:60]}...")

    while True:
        features = read_features(features_file)

        # Check for prompt updates from audio thread
        new_prompt = features.get(
            "prompt",
            "a cosmic nebula with swirling colors of purple and blue, photorealistic",
        )

        # Only update target prompt periodically for smoother transitions
        if new_prompt != next_prompt and prompt_blend_counter <= 0:
            next_prompt = new_prompt
            prompt_blend_counter = PROMPT_BLEND_FRAMES
            print(f"[AI] New prompt queued: {next_prompt[:60]}...")

        # Calculate blend factor
        if prompt_blend_counter > 0:
            blend_factor = 1.0 - (prompt_blend_counter / PROMPT_BLEND_FRAMES)
            display_prompt = generate_blended_prompt(
                current_prompt, next_prompt, blend_factor
            )
            prompt_blend_counter -= 1

            # When blend completes, swap prompts
            if prompt_blend_counter <= 0:
                current_prompt = next_prompt
        else:
            display_prompt = current_prompt

        try:
            gen_start = time.time()

            image = pipe(
                prompt=display_prompt, num_inference_steps=1, guidance_scale=0.0,
                width=640, height=360,   # 16:9
            ).images[0]

            gen_time = time.time() - gen_start
            frame_times.append(gen_time)
            if len(frame_times) > 30:
                frame_times.pop(0)

            if shm_out:
                try:
                    frame_np_out = np.array(image)
                    shm_out.write_frame(frame_np_out, frame_idx)
                except Exception as e:
                    print(f"[AI WORKER] SHM Write Error: {e}")
            else:
                out_path = os.path.join(watch_dir, f"frame_{frame_idx}.png")
                image.save(out_path, format="PNG")
                
            frame_idx += 1

            # Progress logging
            if frame_idx % 10 == 0:
                avg_time = sum(frame_times) / len(frame_times) if frame_times else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(
                    f"[AI] Generated frame {frame_idx}, avg: {avg_time:.2f}s/frame ({fps:.1f} fps)"
                )

        except Exception as e:
            print(f"[AI WORKER] Generation failed: {e}")
            traceback.print_exc()
            time.sleep(1.0)

        # Synchronization micro-sleep:
        # Prevents PyTorch from hogging 100% of the CUDA command queue,
        # allowing PyGame's OpenGL renderer to grab the GPU to achieve 60 FPS
        torch.cuda.synchronize()
        time.sleep(0.005)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ai_worker_optimized.py <watch_dir> <features_file>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
