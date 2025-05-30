import os
import json
import logging
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from PIL import Image

log = logging.getLogger(__name__)

def yolo_to_coco_bbox(
    x_center: float, y_center: float, width: float, height: float,
    img_width: int, img_height: int
) -> List[float]:
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    w = width * img_width
    h = height * img_height
    return [x_min, y_min, w, h]

def parse_yolo_label(
    label_path: str,
    width: int,
    height: int
) -> List[Tuple[int, List[float], float]]:
    annotations = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, w, h = map(float, parts)
                    bbox = yolo_to_coco_bbox(x_center, y_center, w, h, width, height)
                    area = bbox[2] * bbox[3]
                    annotations.append((int(class_id), bbox, area))
                else:
                    log.warning(f"Skipping malformed line in {label_path}: {line.strip()}")
    except Exception as e:
        log.error(f"Failed to parse YOLO label file {label_path}: {e}")
    return annotations

def parse_voc_label(
    label_path: str,
    width: int,
    height: int,
    class_names: List[str]
) -> List[Tuple[int, List[float], float]]:
    annotations = []
    try:
        tree = ET.parse(label_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class_names:
                log.warning(f"Class '{name}' in {label_path} not found in class_names, skipping.")
                continue
            class_id = class_names.index(name)
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            area = w * h
            annotations.append((class_id, bbox, area))
    except Exception as e:
        log.error(f"Failed to parse VOC label file {label_path}: {e}")
    return annotations

def convert_dataset_to_coco(
    images_dir: str,
    labels_dir: str,
    output_json: str,
    label_format: str,
    class_names: List[str]
) -> None:
    if label_format not in ('yolo', 'voc'):
        log.error(f"Unsupported label format: {label_format}. Use 'yolo' or 'voc'.")
        return

    if not os.path.isdir(images_dir):
        log.error(f"Images directory does not exist: {images_dir}")
        return

    if not os.path.isdir(labels_dir):
        log.error(f"Labels directory does not exist: {labels_dir}")
        return

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    images = []
    annotations = []
    annotation_id = 1
    image_id = 1

    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(images_dir, filename)
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            log.error(f"Failed to open image {img_path}: {e}")
            continue

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        label_ext = '.txt' if label_format == 'yolo' else '.xml'
        label_filename = os.path.splitext(filename)[0] + label_ext
        label_path = os.path.join(labels_dir, label_filename)

        if os.path.exists(label_path):
            if label_format == 'yolo':
                parsed = parse_yolo_label(label_path, width, height)
            else:
                parsed = parse_voc_label(label_path, width, height, class_names)

            for class_id, bbox, area in parsed:
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [round(coord, 2) for coord in bbox],
                    "area": round(area, 2),
                    "iscrowd": 0
                })
                annotation_id += 1
        else:
            log.warning(f"Label file not found for image {filename}: {label_path}")

        image_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    try:
        with open(output_json, 'w') as json_file:
            json.dump(coco_format, json_file, indent=4)
        log.info(f"COCO JSON saved to {output_json}")
    except Exception as e:
        log.error(f"Failed to save COCO JSON to {output_json}: {e}")
