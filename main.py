import os
import shutil
import logging
from pathlib import Path

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
DATA_ROOT = r"data/dataset-1_augmentation"

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

CLASS_NAMES = ['Ancylostoma Spp', 'Ascaris Lumbricoides', 'Enterobius Vermicularis', 'Fasciola Hepatica', 'Hymenolepis', 'Schistosoma', 'Taenia Sp', 'Trichuris Trichiura']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

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
def setup_and_prepare_dataset(original_images_dir=None, original_labels_dir=None):
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

    # Copy images if needed
    if not any(Path(IMAGES_DIR).glob("*.[jp][pn]g")) and original_images_dir:
        log.info(f"Copying images from {original_images_dir} to {IMAGES_DIR}")
        for f in os.listdir(original_images_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(os.path.join(original_images_dir, f), IMAGES_DIR)
        log.info("Images copied.")

    # Copy labels if needed
    if not any(Path(LABELS_DIR).glob("*.txt")) and original_labels_dir:
        log.info(f"Copying labels from {original_labels_dir} to {LABELS_DIR}")
        for f in os.listdir(original_labels_dir):
            if f.lower().endswith(".txt"):
                shutil.copy(os.path.join(original_labels_dir, f), LABELS_DIR)
        log.info("Labels copied.")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    try:
        # Provide these if images/labels are not already in input/
        original_images_dir = "original_dataset/images"
        original_labels_dir = "original_dataset/labels"

        setup_and_prepare_dataset(original_images_dir, original_labels_dir)

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
        download_backgrounds(SEARCH_KEYWORD, NUM_BACKGROUNDS, WEBSCRAPE_BG_DIR)

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

        log.info("Pipeline completed successfully!")

    except Exception as e:
        log.exception(f"Pipeline failed: {e}")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
