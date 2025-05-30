import os
import json

def convert_json_to_yolo(json_input_dir, output_labels_dir, class_map):
    os.makedirs(output_labels_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_input_dir) if f.endswith('.json')]

    if not json_files:
        print(f"[INFO] No JSON files found in {json_input_dir}.")
        return

    for json_file in json_files:
        json_path = os.path.join(json_input_dir, json_file)
        print(f"[INFO] Processing {json_path}...")

        with open(json_path, 'r') as f:
            data = json.load(f)

        for image_name, entry in data.items():
            width = entry.get("width")
            height = entry.get("height")
            bboxes = entry.get("bbox", [])

            if not width or not height or not bboxes:
                print(f"[Skip] No valid data for {image_name}")
                continue

            base_name = os.path.splitext(image_name)[0]
            label_path = os.path.join(output_labels_dir, f"{base_name}.txt")
            yolo_lines = []

            for box in bboxes:
                label = box["label"]
                class_id = class_map.get(label, 0)

                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                yolo_lines.append(yolo_line)

            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

            print(f"[Saved] {label_path} ({len(yolo_lines)} objects)")

    print("YOLO annotations generated for all JSON files.")
