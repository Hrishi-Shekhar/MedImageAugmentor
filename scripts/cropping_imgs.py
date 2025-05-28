import os
import cv2
from PIL import Image
import numpy as np

def yolo_to_pixel_bbox(x_center, y_center, width, height, img_width, img_height):
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)
    return xmin, ymin, xmax, ymax

def find_mask_for_image(image_filename):
    name, ext = os.path.splitext(image_filename)
    return f"{name}_superpixels.png"

def crop_using_mask(image_path, mask_path, output_dir, base_name):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Skipping {image_path} - missing image or mask")
        return

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"No object found in {image_path}")
        return

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = image[y:y+h, x:x+w]
        output_filename = f"{base_name}_mask_{i}.jpg"
        cv2.imwrite(os.path.join(output_dir, output_filename), cropped_img)
        print(f"Saved {output_filename}")

def crop_yolo_objects(image_path, label_path, output_dir, class_names):
    img = Image.open(image_path)
    img_width, img_height = img.size

    with open(label_path, 'r') as f:
        lines = f.readlines()

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Skipping malformed line in {label_path}: {line}")
            continue

        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])
        xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(x_center, y_center, w, h, img_width, img_height)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)

        cropped = img.crop((xmin, ymin, xmax, ymax))
        class_name = class_names[class_id]
        output_filename = f"{base_name}_{class_name}_{idx}.jpg"
        cropped.save(os.path.join(output_dir, output_filename))
        print(f"Saved {output_filename}")

def process_dataset(images_dir, labels_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(images_dir):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
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
            print(f"No YOLO label or mask found for {filename}")
