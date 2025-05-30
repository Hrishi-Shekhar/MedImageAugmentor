import os
import random
from PIL import Image

def overlay_foreground_on_background(
    foregrounds_dir,
    backgrounds_dir,
    output_dir,
    lesions_per_image=10
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "composites"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    foreground_files = sorted([f for f in os.listdir(foregrounds_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    background_files = sorted([f for f in os.listdir(backgrounds_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    total_lesions = len(foreground_files)
    lesion_batches = [foreground_files[i:i+lesions_per_image] for i in range(0, total_lesions, lesions_per_image)]

    composite_count = 1

    for bg_file in background_files:
        bg_path = os.path.join(backgrounds_dir, bg_file)
        bg_img = Image.open(bg_path).convert("RGBA")
        bg_width, bg_height = bg_img.size

        print(f"\n[INFO] Processing background: {bg_file}")

        for batch_num, lesion_batch in enumerate(lesion_batches):
            composite = bg_img.copy()
            annotation_lines = []

            for lesion_file in lesion_batch:
                fg_path = os.path.join(foregrounds_dir, lesion_file)
                fg_img = Image.open(fg_path).convert("RGBA")

                # Random scaling
                scale_factor = random.uniform(0.3, 0.6)
                new_w = min(int(fg_img.width * scale_factor), int(bg_width * 0.5))
                new_h = min(int(fg_img.height * scale_factor), int(bg_height * 0.5))
                fg_img = fg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Random position (no overlap constraint for simplicity)
                x = random.randint(0, max(bg_width - fg_img.width, 1))
                y = random.randint(0, max(bg_height - fg_img.height, 1))

                composite.alpha_composite(fg_img, dest=(x, y))

                # YOLO format annotations (class 0 for all)
                x_center = (x + fg_img.width/2)/bg_width
                y_center = (y + fg_img.height/2)/bg_height
                width_norm = fg_img.width/bg_width
                height_norm = fg_img.height/bg_height
                annotation_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

            # Save the composite image
            composite_filename = f"composite_{composite_count}.jpg"
            composite_path = os.path.join(output_dir, "composites", composite_filename)
            composite.convert("RGB").save(composite_path)
            print(f"Saved composite: {composite_path}")

            # Save the annotation file
            annotation_path = os.path.join(output_dir, "annotations", f"composite_{composite_count}.txt")
            with open(annotation_path, "w") as f:
                f.write("\n".join(annotation_lines))

            composite_count += 1

    print("\n All composites and annotations generated successfully!")

