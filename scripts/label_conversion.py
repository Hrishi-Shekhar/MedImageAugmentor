# scripts/label_conversion.py
import os
import json
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

def convert_json_to_yolo(json_dir, output_dir, class_map):
    """
    Convert JSONL annotations (custom format) to YOLO format.

    Args:
        json_dir (str): Path to the JSON annotation folder.
        output_dir (str): Path to save YOLO-format labels.
        class_map (dict): Mapping from class names to IDs.
    """
    if not os.path.exists(json_dir):
        logger.error(f"JSON directory does not exist: {json_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Converting JSON annotations from {json_dir} to YOLO format in {output_dir}...")

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".jsonl")]
    if not json_files:
        logger.warning(f"No JSONL files found in {json_dir}")
        return

    for file in json_files:
        file_path = os.path.join(json_dir, file)
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding error in {file} (line {line_num}): {e}")
                    continue

                filename = data.get("image", {}).get("filename")
                if not filename:
                    logger.warning(f"Skipping line {line_num} in {file} - missing filename key.")
                    continue

                output_label_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
                with open(output_label_path, "w") as out_f:
                    for obj in data.get("objects", []):
                        label = obj.get("label")
                        if label not in class_map:
                            logger.warning(f"Unknown label '{label}' in {file} (line {line_num})")
                            continue

                        class_id = class_map[label]
                        bbox = obj.get("bbox", {})
                        try:
                            xc = float(bbox["x_center"])
                            yc = float(bbox["y_center"])
                            w = float(bbox["width"])
                            h = float(bbox["height"])
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Incomplete or invalid bbox for '{label}' in {file} (line {line_num}): {e}")
                            continue

                        out_f.write(f"{class_id} {xc} {yc} {w} {h}\n")

    logger.info("JSON to YOLO conversion complete.")


def convert_pascal_voc_to_yolo(xml_dir, output_dir, class_names):
    """
    Convert Pascal VOC (XML) annotations to YOLO format.

    Args:
        xml_dir (str): Path to the Pascal VOC XML annotations folder.
        output_dir (str): Path to save YOLO-format labels.
        class_names (list): List of class names (order defines class IDs).
    """
    if not os.path.exists(xml_dir):
        logger.error(f"XML directory does not exist: {xml_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Converting Pascal VOC annotations from {xml_dir} to YOLO format in {output_dir}...")

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if not xml_files:
        logger.warning(f"No XML files found in {xml_dir}")
        return

    for xml_file in xml_files:
        file_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Error parsing XML file {xml_file}: {e}")
            continue

        try:
            img_width = int(root.find("size/width").text)
            img_height = int(root.find("size/height").text)
        except (AttributeError, ValueError) as e:
            logger.warning(f"Missing size info in {xml_file}: {e}")
            continue

        output_label_path = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
        with open(output_label_path, "w") as out_f:
            for obj in root.findall("object"):
                label = obj.find("name").text
                if label not in class_names:
                    logger.warning(f"Skipping unknown label '{label}' in {xml_file}")
                    continue

                class_id = class_names.index(label)
                try:
                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)

                    xc = ((xmin + xmax) / 2) / img_width
                    yc = ((ymin + ymax) / 2) / img_height
                    w = (xmax - xmin) / img_width
                    h = (ymax - ymin) / img_height
                except (AttributeError, ValueError, TypeError) as e:
                    logger.warning(f"Invalid bounding box in {xml_file}: {e}")
                    continue

                out_f.write(f"{class_id} {xc} {yc} {w} {h}\n")

    logger.info("Pascal VOC to YOLO conversion complete.")
