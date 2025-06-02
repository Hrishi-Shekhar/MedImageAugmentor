import cv2
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)

def yolo_to_masks(
    images_dir: str | Path,
    labels_dir: str | Path,
    masks_dir: str | Path,
    multi_class: bool = False
) -> None:
    """
    Convert YOLO annotations to mask images.

    Args:
        images_dir (str or Path): Directory containing input images.
        labels_dir (str or Path): Directory containing YOLO label files.
        masks_dir (str or Path): Directory where masks will be saved.
        multi_class (bool): If True, use class IDs for mask values. If False, use binary masks.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing mask images
    existing_masks = list(masks_dir.glob("*.png"))
    for mask_file in existing_masks:
        try:
            mask_file.unlink()
            log.debug(f"Deleted old mask: {mask_file}")
        except Exception as e:
            log.warning(f"Could not delete {mask_file}: {e}")

    image_files = list(images_dir.glob("*.[jp][pn]g"))  # jpg, jpeg, png

    if not image_files:
        log.warning(f"No images found in {images_dir}")
        return

    for image_file in image_files:
        label_file = labels_dir / f"{image_file.stem}.txt"
        mask_file = masks_dir / f"{image_file.stem}.png"

        img = cv2.imread(str(image_file))
        if img is None:
            log.warning(f"Could not read image: {image_file}")
            continue
        h, w, _ = img.shape

        mask = np.zeros((h, w), dtype=np.uint8)

        if not label_file.exists():
            log.warning(f"No label found for {image_file.name}")
            cv2.imwrite(str(mask_file), mask)
            continue

        with label_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    log.warning(f"Invalid YOLO label format in {label_file}")
                    continue
                try:
                    cls, x_center, y_center, width, height = map(float, parts)
                    cls_id = int(cls) + 1 if multi_class else 255

                    # Convert to pixel coordinates
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h

                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w - 1, x_max)
                    y_max = min(h - 1, y_max)

                    mask[y_min:y_max, x_min:x_max] = cls_id

                except ValueError:
                    log.warning(f"Skipping line due to conversion error in {label_file}")
                    continue

        cv2.imwrite(str(mask_file), mask)

    log.info(f"Masks generated in: {masks_dir}")

