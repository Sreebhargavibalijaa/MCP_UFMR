
import numpy as np
from PIL import Image
import os
import time

def plot_patch_overlay_on_image(patch_tensor, h, w, image_tensor):
    heatmap = (patch_tensor.numpy() * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize((224, 224))
    timestamp = str(int(time.time()))
    path = f"overlay_{timestamp}.png"
    heatmap_img.save(path)
    return path
