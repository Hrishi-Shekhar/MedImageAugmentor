from scripts.cropping_imgs import process_dataset
from scripts.bg_removal import remove_bg_batch
from scripts.bg_extraction_web_scraping import download_backgrounds
from scripts.overlay import overlay_foreground_on_background
import os

if __name__ == "__main__":

    images_dir = r"data/Skin_cancer_augmentation/input/images"
    labels_dir = r"data/Skin_cancer_augmentation/input/labels"
    cropped_dir = r"data/Skin_cancer_augmentation/intermediate/cropped"

    class_names = ["negative", "positive"]

    process_dataset(images_dir, labels_dir, cropped_dir, class_names)

    cropped_nobg_dir = r"data/Skin_cancer_augmentation/intermediate/cropped_nobg"

    remove_bg_batch(cropped_dir,cropped_nobg_dir)

    search_keyword = "medical skin background, macro skin surface, dermatology texture, neutral pinkish skin, no moles or lesions, high-quality close-up"
    background_folder = r"data/Skin_cancer_augmentation/backgrounds/Web_scraping"
    num_images = 20

    download_backgrounds(search_keyword, num_images, background_folder)

    overlay_foreground_on_background(
        cropped_nobg_dir,
        backgrounds_dir= os.path.join(background_folder,search_keyword),
        output_dir=r"data/Skin_cancer_augmentation/output/composites"
    )