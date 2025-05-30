from scripts.cropping_imgs import process_dataset
from scripts.bg_removal import remove_bg_batch
from scripts.bg_extraction_web_scraping import download_backgrounds
from scripts.overlay import overlay_foreground_on_background
from scripts.json_to_yolo import convert_json_to_yolo  # Import your JSON-to-YOLO function
from scripts.yolo_to_json import convert_dataset_to_coco  # Import your YOLO-to-COCO function
import os
import glob

if __name__ == "__main__":

    images_dir = r"data/Skin_cancer_augmentation/input/images"
    labels_dir = r"data/Skin_cancer_augmentation/input/labels"
    cropped_dir = r"data/Skin_cancer_augmentation/intermediate/cropped"

    class_names = ['polyp']
    class_map = {name: idx for idx, name in enumerate(class_names)}

    # Check if JSON annotations are available
    json_input_dir = r"data/Skin_cancer_augmentation/input/json_input"
    if os.path.exists(json_input_dir) and any(f.endswith(".json") for f in os.listdir(json_input_dir)):
        print("[INFO] JSON annotations found. Converting to YOLO format...")
        convert_json_to_yolo(json_input_dir, labels_dir, class_map)
    else:
        print("[INFO] No JSON annotations found. Skipping JSON to YOLO conversion.")

    # Step 1: Crop images based on YOLO annotations
    process_dataset(images_dir, labels_dir, cropped_dir, class_names)

    # Step 2: Background removal
    cropped_nobg_dir = r"data/Skin_cancer_augmentation/intermediate/cropped_nobg"
    remove_bg_batch(cropped_dir, cropped_nobg_dir)

    # Step 3: Download backgrounds
    search_keyword = "macro close-up of human internal organ textures under endoscopy"
    background_folder = r"data/Skin_cancer_augmentation/backgrounds/Web_scraping"
    num_images = 20

    download_backgrounds(search_keyword, num_images, background_folder)

    # Step 4: Overlay foreground on backgrounds
    overlay_foreground_on_background(
        cropped_nobg_dir,
        backgrounds_dir=os.path.join(background_folder, search_keyword),
        output_dir=r"data/Skin_cancer_augmentation/output"
    )

    # Step 5: Convert YOLO labels to COCO format
    images_dir = 'data/Skin_cancer_augmentation/input/images'
    labels_dir = 'data/Skin_cancer_augmentation/input/labels'  # or 'xml'
    output_json = 'data/Skin_cancer_augmentation/output/coco_annotations.json'
    label_format = 'yolo'  # or 'voc'

    # Run conversion
    convert_dataset_to_coco(images_dir, labels_dir, output_json, label_format, class_names)
