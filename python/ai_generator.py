import torch
from diffusers import StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from susa import NEGATIVE_PROMPT


class AIGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", width=640, height=360):
        self.device = device
        self.width = width
        self.height = height

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(device)

        self.stream = StreamDiffusion(
            pipe=pipe,
            use_denoising_batch=True,
            t_index_list=[0, 1, 2, 3],
            frame_buffer_size=1,
            cfg_type="none",
            width=self.width,
            height=self.height,
        )
        self.stream.load_lcm_lora()
        self.stream.fuse_lora()
        print("TensorRT is not available, running without it.")

    def generate_image(self, prompt):
        try:
            self.stream.prepare(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=4,
            )
            for _ in range(self.stream.batch_size - 1):
                self.stream()
            output_tensor = self.stream()
            pil_image = postprocess_image(output_tensor, output_type="pil")[0]
            return pil_image.convert("RGB")
        except torch.cuda.OutOfMemoryError:
            print("ERROR: CUDA out of memory during image generation.")
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"ERROR: Image generation failed: {e}")
            raise


if __name__ == '__main__':
    generator = AIGenerator()
    from susa import Susa
    susa = Susa()
    prompt = susa.generate_prompt(2000)
    print(f"Test prompt: {prompt}")
    image = generator.generate_image(prompt)
    image.save("output.png")
    print("Image saved to output.png")
