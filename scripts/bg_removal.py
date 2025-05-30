from rembg import remove
import os
import shutil
import logging
from pathlib import Path

log = logging.getLogger(__name__)

def remove_bg_batch(input_folder, output_folder):
    """
    Remove backgrounds from all images in input_folder and save to output_folder.
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
            with output_file.open('wb') as output_f:
                output_f.write(output_data)
            log.info(f"Processed: {image_file.name}")
        except Exception as e:
            log.error(f"Failed to process {image_file.name}: {e}")

