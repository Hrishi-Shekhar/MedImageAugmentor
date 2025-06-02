import os
import logging
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

# Background folders
BACKGROUND_ROOT = os.path.join(DATA_ROOT, "backgrounds")
WEBSCRAPE_BG_DIR = os.path.join(BACKGROUND_ROOT, "Web_scraping")
USER_BG_DIR = os.path.join(BACKGROUND_ROOT, "user_generated_bgs")

# Output folders
OUTPUT_ROOT = os.path.join(DATA_ROOT, "output")
COMPOSITES_DIR = os.path.join(OUTPUT_ROOT,"composites")
ANNOTATIONS_DIR = os.path.join(OUTPUT_ROOT,"annotations")
COCO_JSON_PATH = os.path.join(OUTPUT_ROOT, "coco_annotations.json")
MASKS_DIR = os.path.join(OUTPUT_ROOT,"masks")

# Annotation sources
JSON_ANNOTATIONS_DIR = os.path.join(DATA_ROOT,"input","json_input")
XML_ANNOTATIONS_DIR = os.path.join(DATA_ROOT,"input","xml_input")

# Classes
#CLASS_NAMES = ["apple", "banana", "orange"]
CLASS_NAMES = ['Ancylostoma Spp', 'Ascaris Lumbricoides', 'Enterobius Vermicularis', 'Fasciola Hepatica', 'Hymenolepis', 'Schistosoma', 'Taenia Sp', 'Trichuris Trichiura']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Background Search
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
# MAIN PIPELINE
# -----------------------------
def main():
    try:
        # Step 1: Convert Annotations (JSON or XML) -> YOLO
        if os.path.exists(JSON_ANNOTATIONS_DIR) and any(f.endswith(".jsonl") for f in os.listdir(JSON_ANNOTATIONS_DIR)):
            log.info("JSON annotations found. Converting to YOLO format...")
            convert_json_to_yolo(JSON_ANNOTATIONS_DIR, LABELS_DIR, CLASS_MAP)
        elif os.path.exists(XML_ANNOTATIONS_DIR) and any(f.endswith(".xml") for f in os.listdir(XML_ANNOTATIONS_DIR)):
            log.info("XML annotations found. Converting to YOLO format...")
            convert_pascal_voc_to_yolo(XML_ANNOTATIONS_DIR, LABELS_DIR, CLASS_NAMES)
        else:
            log.warning("No JSON or XML annotations found. Skipping annotation conversion.")

        # Step 2: Crop images based on YOLO annotations
        log.info("Cropping images based on YOLO annotations...")
        process_dataset(IMAGES_DIR, LABELS_DIR, CROPPED_DIR, CLASS_NAMES)

        # Step 3: Background removal
        log.info("Removing backgrounds from cropped images...")
        remove_bg_batch(CROPPED_DIR, CROPPED_NOBG_DIR)

        # Step 4: Download backgrounds
        log.info(f"Downloading {NUM_BACKGROUNDS} background images for search: '{SEARCH_KEYWORD}'")
        download_backgrounds(SEARCH_KEYWORD, NUM_BACKGROUNDS, WEBSCRAPE_BG_DIR)

        # Step 5: Overlay foregrounds on backgrounds

        # Use Web-scraped + User-provided backgrounds
        for bg_source in [os.path.join(WEBSCRAPE_BG_DIR, SEARCH_KEYWORD), USER_BG_DIR]:
            if not os.path.exists(bg_source):
                log.warning(f"Background folder not found: {bg_source}. Skipping...")
                continue

            log.info(f"Overlaying cropped foregrounds onto backgrounds in: {bg_source}")
            overlay_foreground_on_background(
                foregrounds_dir=CROPPED_NOBG_DIR,
                backgrounds_dir=bg_source,
                composites_dir=COMPOSITES_DIR,
                annotations_dir=ANNOTATIONS_DIR,
                class_names=CLASS_NAMES
            )

        # Step 6: Convert output YOLO annotations to COCO format
        log.info("Converting YOLO annotations to COCO format...")
        convert_dataset_to_coco(
            images_dir=COMPOSITES_DIR,
            labels_dir=ANNOTATIONS_DIR,
            output_json=COCO_JSON_PATH,
            label_format="yolo",
            class_names=CLASS_NAMES
        )

        # Step 7: Convert output YOLO annotations to masks
        log.info("Converting YOLO annotations to masks...")
        yolo_to_masks(
            images_dir=COMPOSITES_DIR,
            labels_dir=ANNOTATIONS_DIR,
            masks_dir=MASKS_DIR,
        )

        log.info("Pipeline completed successfully!")

    except Exception as e:
        log.exception(f"Pipeline failed with error: {e}")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
