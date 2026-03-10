import os
import pygame
from shader_renderer import ShaderRenderer
from PIL import Image
import numpy as np

def main():
    print("Initializing ShaderRenderer...")
    try:
        renderer = ShaderRenderer(640, 360)
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    img = Image.new("RGB", (640, 360), (100, 150, 200))
    print("Testing first render...")
    try:
        renderer.render(img)
        print("First render success!")
    except Exception as e:
        print(f"RENDER FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
