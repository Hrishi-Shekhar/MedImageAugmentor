import os
import random
from PIL import Image

def overlay_foreground_on_background(
    foregrounds_dir,
    backgrounds_dir,
    output_dir,
    num_augmented_images=25,
    min_foregrounds_per_bg=2,
    max_foregrounds_per_bg=4
):
    os.makedirs(output_dir, exist_ok=True)

    foreground_files = [f for f in os.listdir(foregrounds_dir) if f.endswith(('.png', '.jpg'))]
    background_files = [f for f in os.listdir(backgrounds_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def check_overlap(box1, box2, margin=10):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max + margin < x2_min or x1_min - margin > x2_max or
                    y1_max + margin < y2_min or y1_min - margin > y2_max)

    for i in range(num_augmented_images):
        bg_file = random.choice(background_files)
        bg_path = os.path.join(backgrounds_dir, bg_file)
        bg_img = Image.open(bg_path).convert("RGBA")
        bg_width, bg_height = bg_img.size

        composite = bg_img.copy()
        placed_boxes = []

        num_foregrounds_per_bg = random.randint(min_foregrounds_per_bg, max_foregrounds_per_bg)

        for _ in range(num_foregrounds_per_bg):
            fg_file = random.choice(foreground_files)
            fg_path = os.path.join(foregrounds_dir, fg_file)
            fg_img = Image.open(fg_path).convert("RGBA")

            scale_factor = random.uniform(0.3, 0.6)
            new_w = min(int(fg_img.width * scale_factor), int(bg_width * 0.5))
            new_h = min(int(fg_img.height * scale_factor), int(bg_height * 0.5))
            fg_img = fg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            max_attempts = 50
            for _ in range(max_attempts):
                x = random.randint(0, max(bg_width - fg_img.width, 1))
                y = random.randint(0, max(bg_height - fg_img.height, 1))
                new_box = (x, y, x + fg_img.width, y + fg_img.height)

                if not any(check_overlap(new_box, b) for b in placed_boxes):
                    placed_boxes.append(new_box)
                    composite.alpha_composite(fg_img, dest=(x, y))
                    break
            else:
                print(f"Could not place {fg_file} without overlap after {max_attempts} attempts.")

        output_path = os.path.join(output_dir, f"composite_{i+1}.jpg")
        composite.convert("RGB").save(output_path)
        print(f"Saved: {output_path}")


