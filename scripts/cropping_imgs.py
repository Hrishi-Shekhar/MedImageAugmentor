import os
import cv2
from PIL import Image
import shutil
import logging
from typing import List, Tuple
import yaml

def load_config(path = "config.yaml"):
    with open(path,'r') as f:
        return yaml.safe_load(f)

log = logging.getLogger(__name__)

config = load_config()

def yolo_to_pixel_bbox(
    x_center: float, y_center: float, width: float, height: float,
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)

    return xmin, ymin, xmax, ymax

def find_mask_for_image(image_filename: str) -> str:
    name, ext = os.path.splitext(image_filename)
    return f"{name}_superpixels.png"

def crop_using_mask(
    image_path: str, mask_path: str, output_dir: str, base_name: str, min_size: Tuple[int,int] = config["cropping"]["min_size"]
) -> None:
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        log.warning(f"Skipping {image_path} - missing image or mask")
        return

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.warning(f"No objects found in mask for {image_path}")
        return

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = image[y:y + h, x:x + w]
        if cropped_img.shape[0] < min_size[1] or cropped_img[1] < min_size[0]:
            log.info(f"Skipping small crop ({cropped_img.shape[1]},{cropped_img.shape[0]}) from {image_path}")
            continue
        output_filename = f"{base_name}_mask_{i}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped_img)
        log.info(f"Saved {output_path}")

def crop_yolo_objects(
    image_path: str, label_path: str, output_dir: str, class_names: List[str], min_size: Tuple[int,int] = config['cropping']['min_size']
) -> None:
    img = Image.open(image_path)
    img_width, img_height = img.size

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            log.warning(f"Skipping malformed line in {label_path}: {line.strip()}")
            continue

        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])
        xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(x_center, y_center, w, h, img_width, img_height)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)

        cropped = img.crop((xmin, ymin, xmax, ymax))
        if cropped.width < min_size[0] or cropped.height < min_size[1]:
            log.info(f"Skipping small crop ({cropped.width},{cropped.height}) from {image_path}")
            continue
        class_name = class_names[class_id]
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_mask_{idx}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cropped.save(output_path)
        log.info(f"Saved {output_path}")

def process_dataset(
    images_dir: str, labels_dir: str, output_dir: str, class_names: List[str]
) -> None:
    if os.path.exists(output_dir):
        log.info(f"Clearing previous outputs in {output_dir}...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
        mask_path = os.path.join(images_dir, find_mask_for_image(filename))
        base_name = os.path.splitext(filename)[0]

        if os.path.exists(label_path):
            crop_yolo_objects(image_path, label_path, output_dir, class_names)
        elif os.path.exists(mask_path):
            crop_using_mask(image_path, mask_path, output_dir, base_name)
        else:
            log.warning(f"No YOLO label or mask found for {filename}")
