import os
import random
import logging
from typing import List
from PIL import Image

log = logging.getLogger(__name__)

def ensure_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    return sorted([
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in extensions
    ])

def find_class_id(filename: str, class_names: List[str]) -> int:
    lower_name = filename.lower()
    for idx, class_name in enumerate(class_names):
        if class_name.lower() in lower_name:
            return idx
    log.warning(f"Class not found in filename '{filename}', defaulting to class 0")
    return 0

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def overlay_foreground_on_background(
    foregrounds_dir: str,
    backgrounds_dir: str,
    composites_dir: str,
    annotations_dir: str,
    class_names: List[str],
    lesions_per_image: int = 4,
    max_attempts: int = 20,
) -> None:
    ensure_dir(composites_dir)
    ensure_dir(annotations_dir)

    foreground_files = get_image_files(foregrounds_dir)
    background_files = get_image_files(backgrounds_dir)

    if not foreground_files:
        log.error("No foreground images found in %s", foregrounds_dir)
        return
    if not background_files:
        log.error("No background images found in %s", backgrounds_dir)
        return

    lesion_batches = [foreground_files[i:i + lesions_per_image] for i in range(0, len(foreground_files), lesions_per_image)]
    composite_count = 1

    for bg_file in background_files:
        bg_path = os.path.join(backgrounds_dir, bg_file)
        bg_img = Image.open(bg_path).convert("RGBA")
        bg_width, bg_height = bg_img.size
        log.info("Processing background: %s", bg_file)

        for batch_num, lesion_batch in enumerate(lesion_batches):
            composite = bg_img.copy()
            annotation_lines = []
            occupied_boxes = []

            placed_count = 0
            for fg_file in lesion_batch:
                fg_path = os.path.join(foregrounds_dir, fg_file)
                fg_img = Image.open(fg_path).convert("RGBA")
                fg_width, fg_height = fg_img.size

                if fg_width > bg_width or fg_height > bg_height:
                    log.warning("Skipping %s because it is larger than the background.", fg_file)
                    continue

                class_id = find_class_id(fg_file, class_names)

                placed = False
                for _ in range(max_attempts):
                    x = random.randint(0, max(bg_width - fg_width, 0))
                    y = random.randint(0, max(bg_height - fg_height, 0))
                    new_box = (x, y, x + fg_width, y + fg_height)

                    overlap = any(calculate_iou(new_box, box) > 0.1 for box in occupied_boxes)
                    if not overlap:
                        occupied_boxes.append(new_box)
                        placed = True
                        break

                if not placed:
                    log.warning("Could not place %s without excessive overlap after %d attempts", fg_file, max_attempts)
                    continue

                composite.alpha_composite(fg_img, dest=(x, y))

                x_center = (x + fg_width / 2) / bg_width
                y_center = (y + fg_height / 2) / bg_height
                width_norm = fg_width / bg_width
                height_norm = fg_height / bg_height
                annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
                placed_count += 1

            if placed_count == 0:
                log.info("No lesions placed on background %s, skipping composite.", bg_file)
                continue

            composite_filename = f"composite_{composite_count}.jpg"
            composite_path = os.path.join(composites_dir, composite_filename)
            composite.convert("RGB").save(composite_path)
            log.info("Saved composite: %s", composite_path)

            annotation_path = os.path.join(annotations_dir, f"composite_{composite_count}.txt")
            with open(annotation_path, 'w') as f:
                f.write("\n".join(annotation_lines))
            composite_count += 1

    log.info("All composites and annotations generated successfully!")
