import os
import shutil
import logging
from pathlib import Path
import numpy as np
from PIL import Image

from scripts.cropping_imgs import process_dataset
from scripts.bg_removal import remove_bg_batch
from scripts.bg_extraction_web_scraping import download_backgrounds
from scripts.overlay import overlay_foreground_on_background
from scripts.label_conversion import convert_json_to_yolo, convert_pascal_voc_to_yolo
from scripts.yolo_to_json import convert_dataset_to_coco
from scripts.yolo_to_mask import yolo_to_masks

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = r"C:\Users\hrish\Desktop\dataset_001"

IMAGES_DIR = os.path.join(DATA_ROOT, "input", "images")
LABELS_DIR = os.path.join(DATA_ROOT, "input", "labels")
CROPPED_DIR = os.path.join(DATA_ROOT, "intermediate", "cropped")
CROPPED_NOBG_DIR = os.path.join(DATA_ROOT, "intermediate", "cropped_nobg")

BACKGROUND_ROOT = os.path.join(DATA_ROOT, "backgrounds")
WEBSCRAPE_BG_DIR = os.path.join(BACKGROUND_ROOT, "web_scraping")
USER_BG_DIR = os.path.join(BACKGROUND_ROOT, "user_generated")

OUTPUT_ROOT = os.path.join(DATA_ROOT, "output")
COMPOSITES_DIR = os.path.join(OUTPUT_ROOT, "composites")
ANNOTATIONS_DIR = os.path.join(OUTPUT_ROOT, "annotations")
COCO_JSON_PATH = os.path.join(OUTPUT_ROOT, "coco_annotations.json")
MASKS_DIR = os.path.join(OUTPUT_ROOT, "masks")

JSON_ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "input", "json_input")
XML_ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "input", "xml_input")

# CLASS_NAMES = os.path.join(DATA_ROOT,"class_names")
# CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SEARCH_KEYWORD = "A high-resolution sterile laboratory background, with subtle gradients and smooth textures, softly illuminated under brightfield microscopy."
NUM_BACKGROUNDS = 20

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s | %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# -----------------------------
# SETUP: Ensure folders exist and data is moved if needed
# -----------------------------

def get_average_image_dimensions(images_dir):
    widths, heights = [], []
    for img_file in Path(images_dir).glob("*.[jp][pn]g"):
        try:
            with Image.open(img_file) as img:
                widths.append(img.width)
                heights.append(img.height)
        except Exception as e:
            log.warning(f"Skipping {img_file}: {e}")
    avg_width = int(np.mean(widths)) if widths else 512
    avg_height = int(np.mean(heights)) if heights else 512
    log.info(f"Average image dimensions: {avg_width}x{avg_height}")
    return avg_width, avg_height

def setup_and_prepare_dataset(original_images_dir=None, original_labels_dir=None, original_class_name_file=None, original_test_dir=None):
    required_dirs = [
        "data",
        DATA_ROOT,
        IMAGES_DIR,
        LABELS_DIR,
        CROPPED_DIR,
        CROPPED_NOBG_DIR,
        os.path.join(WEBSCRAPE_BG_DIR, SEARCH_KEYWORD),
        USER_BG_DIR,
        COMPOSITES_DIR,
        ANNOTATIONS_DIR,
        MASKS_DIR
    ]

    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
        log.info(f"Ensured directory exists: {d}")

    # Copy images
    if not any(Path(IMAGES_DIR).glob("*.[jp][pn]g")) and original_images_dir:
        log.info(f"Copying images from {original_images_dir} to {IMAGES_DIR}")
        for f in os.listdir(original_images_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(os.path.join(original_images_dir, f), IMAGES_DIR)
        log.info("Images copied.")

    # Copy labels
    if not any(Path(LABELS_DIR).glob("*.txt")) and original_labels_dir:
        log.info(f"Copying labels from {original_labels_dir} to {LABELS_DIR}")
        for f in os.listdir(original_labels_dir):
            if f.lower().endswith(".txt"):
                shutil.copy(os.path.join(original_labels_dir, f), LABELS_DIR)
        log.info("Labels copied.")

    # Copy class names
    if original_class_name_file:
        input_class_names = os.path.join(DATA_ROOT, "input", "class_names.txt")
        if not os.path.exists(input_class_names):
            shutil.copy(original_class_name_file, input_class_names)
            log.info(f"Copied class names to {input_class_names}")

    input_test_dir = os.path.join(DATA_ROOT, "input", "test")
    if original_test_dir and os.path.exists(original_test_dir):
        log.info(f"Copying test data from {original_test_dir} to {input_test_dir}")
        shutil.copytree(original_test_dir, input_test_dir, dirs_exist_ok=True)
        log.info("Test data copied.")

    # Remove originals after copying
    if original_images_dir and os.path.exists(original_images_dir):
        shutil.rmtree(original_images_dir)
        log.info(f"Removed original images directory: {original_images_dir}")

    if original_labels_dir and os.path.exists(original_labels_dir):
        shutil.rmtree(original_labels_dir)
        log.info(f"Removed original labels directory: {original_labels_dir}")

    if original_class_name_file and os.path.exists(original_class_name_file):
        os.remove(original_class_name_file)
        log.info(f"Removed original class names file: {original_class_name_file}")

    if original_test_dir and os.path.exists(original_test_dir):
        shutil.rmtree(original_test_dir)
        log.info(f"Removed original test directory: {original_test_dir}")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    try:
        # Provide these if images/labels are not already in input/
        original_images_dir = os.path.join(DATA_ROOT,"images")
        original_labels_dir = os.path.join(DATA_ROOT,"labels")
        original_test_dir = os.path.join(DATA_ROOT,"test")
        original_class_names_file = os.path.join(DATA_ROOT,"class_names.txt")
        CLASS_NAMES = os.path.join(DATA_ROOT, "input", "class_names.txt")
        CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


        setup_and_prepare_dataset(original_images_dir, original_labels_dir, original_class_names_file, original_test_dir)

        # Step 1: Convert annotations
        if os.path.exists(JSON_ANNOTATIONS_DIR) and any(f.endswith(".jsonl") for f in os.listdir(JSON_ANNOTATIONS_DIR)):
            log.info("Converting JSON annotations to YOLO format...")
            convert_json_to_yolo(JSON_ANNOTATIONS_DIR, LABELS_DIR, CLASS_MAP)
        elif os.path.exists(XML_ANNOTATIONS_DIR) and any(f.endswith(".xml") for f in os.listdir(XML_ANNOTATIONS_DIR)):
            log.info("Converting XML annotations to YOLO format...")
            convert_pascal_voc_to_yolo(XML_ANNOTATIONS_DIR, LABELS_DIR, CLASS_NAMES)
        else:
            log.warning("No JSON or XML annotations found. Skipping annotation conversion.")

        # Step 2: Crop images
        log.info("Cropping objects from input images...")
        process_dataset(IMAGES_DIR, LABELS_DIR, CROPPED_DIR, CLASS_NAMES)

        # Step 3: Background removal
        log.info("Removing background from cropped images...")
        remove_bg_batch(CROPPED_DIR, CROPPED_NOBG_DIR)

        # Step 4: Background download
        log.info(f"Downloading {NUM_BACKGROUNDS} backgrounds for: {SEARCH_KEYWORD}")
        avg_w, avg_h = get_average_image_dimensions(IMAGES_DIR)
        download_backgrounds(
            keyword=SEARCH_KEYWORD,
            limit=NUM_BACKGROUNDS,
            output_dir=WEBSCRAPE_BG_DIR,
            width=avg_w,
            height=avg_h
        )

        # Step 5: Overlay
        for bg_source in [os.path.join(WEBSCRAPE_BG_DIR, SEARCH_KEYWORD), USER_BG_DIR]:
            if os.path.exists(bg_source):
                log.info(f"Overlaying foregrounds on backgrounds from: {bg_source}")
                overlay_foreground_on_background(
                    foregrounds_dir=CROPPED_NOBG_DIR,
                    backgrounds_dir=bg_source,
                    composites_dir=COMPOSITES_DIR,
                    annotations_dir=ANNOTATIONS_DIR,
                    class_names=CLASS_NAMES
                )

        # Step 6: Convert to COCO
        log.info("Converting YOLO annotations to COCO format...")
        convert_dataset_to_coco(
            images_dir=COMPOSITES_DIR,
            labels_dir=ANNOTATIONS_DIR,
            output_json=COCO_JSON_PATH,
            label_format="yolo",
            class_names=CLASS_NAMES
        )

        # Step 7: Generate masks
        log.info("Generating masks from YOLO annotations...")
        yolo_to_masks(COMPOSITES_DIR, ANNOTATIONS_DIR, MASKS_DIR)

        # Copy original images to composites folder
        for img_file in Path(IMAGES_DIR).glob("*.[jp][pn]g"):
            dst = Path(COMPOSITES_DIR) / f"orig_{img_file.name}"
            shutil.copy(img_file, dst)
        log.info("Original images copied to composites folder.")

        # Copy original labels to annotations folder
        for label_file in Path(LABELS_DIR).glob("*.txt"):
            dst = Path(ANNOTATIONS_DIR) / f"orig_{label_file.name}"
            shutil.copy(label_file, dst)
        log.info("Original labels copied to annotations folder.")

        # Generate masks for original images too
        temp_mask_dir = os.path.join(DATA_ROOT, "intermediate", "orig_masks")
        os.makedirs(temp_mask_dir, exist_ok=True)
        yolo_to_masks(IMAGES_DIR, LABELS_DIR, temp_mask_dir)

        # Copy masks to final output
        for mask_file in Path(temp_mask_dir).glob("*.png"):
            dst = Path(MASKS_DIR) / f"orig_{mask_file.name}"
            shutil.copy(mask_file, dst)
        log.info("Original masks copied to masks folder.")

        log.info("Pipeline completed successfully!")


    except Exception as e:
        log.exception(f"Pipeline failed: {e}")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()