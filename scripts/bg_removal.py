from rembg import remove, new_session
import os
import shutil
import logging
from pathlib import Path
from PIL import Image
import io
import numpy as np
import yaml

def load_yaml(path = 'config.yaml'):
    with open(path,'r') as f:
        return yaml.safe_load(f)
    
config = load_yaml()

log = logging.getLogger(__name__)

def is_image_significant(image_bytes, min_foreground_pixels):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(img.split()[-1])
    non_zero = np.count_nonzero(alpha > config["bg_removal"]["min_alpha"])
    return non_zero >= min_foreground_pixels


def remove_bg_batch(input_folder, output_folder):
    """
    Remove backgrounds from all images in input_folder using the isnet-general-use model.
    """
    output_folder = Path(output_folder)
    input_folder = Path(input_folder)
    
    if output_folder.exists():
        log.info(f"Clearing previous outputs in {output_folder}...")
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)


    for image_file in input_folder.iterdir():
        if image_file.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue

        output_file = output_folder / (image_file.stem + "_no_bg.png")
        try:
            with image_file.open('rb') as input_f:
                input_data = input_f.read()
                output_data = remove(input_data)  

            if not is_image_significant(output_data,min_foreground_pixels=config["bg_removal"]["min_foreground_pixels"]):
                log.info(f"Skipped (too small or empty): {image_file.name}")
                continue

            with output_file.open('wb') as output_f:
                output_f.write(output_data)
            log.info(f"Processed: {image_file.name}")
        except Exception as e:
            log.error(f"Failed to process {image_file.name}: {e}")
