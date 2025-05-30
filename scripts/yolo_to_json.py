import os
import json
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

def yolo_to_coco_bbox(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    w = width * img_width
    h = height * img_height
    return [x_min, y_min, w, h]

def parse_yolo_label(label_path, width, height):
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, w, h = map(float, parts)
                bbox = yolo_to_coco_bbox(x_center, y_center, w, h, width, height)
                area = bbox[2] * bbox[3]
                annotations.append((int(class_id), bbox, area))
    return annotations

def parse_voc_label(label_path, width, height, class_names):
    annotations = []
    tree = ET.parse(label_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        class_id = class_names.index(name) + 1 if name in class_names else None
        if class_id:
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            area = w * h
            annotations.append((class_id - 1, bbox, area))
    return annotations

def convert_dataset_to_coco(images_dir, labels_dir, output_json, label_format, class_names):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    images = []
    annotations = []
    annotation_id = 1
    image_id = 1

    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(images_dir, filename)
        img = Image.open(img_path)
        width, height = img.size

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        label_filename = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_dir, label_filename + ('.txt' if label_format == 'yolo' else '.xml'))

        if os.path.exists(label_path):
            if label_format == 'yolo':
                parsed = parse_yolo_label(label_path, width, height)
            elif label_format == 'voc':
                parsed = parse_voc_label(label_path, width, height, class_names)
            else:
                parsed = []

            for class_id, bbox, area in parsed:
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [round(coord, 2) for coord in bbox],
                    "area": round(area, 2),
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    coco_format = {"images": images, "annotations": annotations, "categories": categories}

    with open(output_json, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)
    print(f"COCO JSON saved to {output_json}")

