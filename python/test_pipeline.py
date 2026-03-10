"""Quick standalone test: does SD 1.5 + LCM-LoRA img2img run >10 frames?"""
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForImage2Image, LCMScheduler
import time

MODEL_ID    = "runwayml/stable-diffusion-v1-5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
GEN_W, GEN_H = 512, 288

print("Loading pipeline...")
pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.load_lora_weights(LCM_LORA_ID)
pipe.fuse_lora()
pipe.enable_attention_slicing()
print("Pipeline ready.")

frame = Image.fromarray(np.random.randint(32, 224, (GEN_H, GEN_W, 3), dtype=np.uint8))
prompt = "neon abstract fluid art, vibrant colors"

for i in range(20):
    t0 = time.time()
    print(f"Frame {i}...", flush=True)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt, image=frame, strength=0.48,
            num_inference_steps=4, guidance_scale=1.0,
            width=GEN_W, height=GEN_H,
        )
    frame = result.images[0]
    print(f"Frame {i} done in {time.time()-t0:.2f}s", flush=True)

print("All 20 frames done!")
