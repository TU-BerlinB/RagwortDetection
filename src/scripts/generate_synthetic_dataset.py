import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import tqdm

def create_synthetic_dataset(
    bg_dir="dataset_builder/backgrounds",
    fg_dir="dataset_builder/foregrounds",
    output_dir="datasets/synthetic_ragwort",
    num_images=1000,
    img_size=(640, 640),
    max_plants_per_image=3
):
    # Setup directories
    bg_path = Path(bg_dir)
    fg_path = Path(fg_dir)
    
    out_images = Path(output_dir) / "images" / "train"
    out_labels = Path(output_dir) / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Load file paths
    bg_files = list(bg_path.glob("*.jpg")) + list(bg_path.glob("*.png")) + list(bg_path.glob("*.jpeg"))
    fg_files = list(fg_path.glob("*.png"))  # Foregrounds MUST be PNGs with transparency

    if not bg_files:
        print(f"Error: No background images found in {bg_dir}")
        return
    if not fg_files:
        print(f"Error: No foreground images (transparent PNGs) found in {fg_dir}")
        return

    print(f"Found {len(bg_files)} backgrounds and {len(fg_files)} foregrounds.")
    print(f"Generating {num_images} synthetic images...")

    for i in tqdm.tqdm(range(num_images)):
        # 1. Prepare background
        bg_file = random.choice(bg_files)
        bg = Image.open(bg_file).convert("RGBA")
        
        # Randomly crop the background to the target size
        if bg.width > img_size[0] and bg.height > img_size[1]:
            x_offset = random.randint(0, bg.width - img_size[0])
            y_offset = random.randint(0, bg.height - img_size[1])
            bg = bg.crop((x_offset, y_offset, x_offset + img_size[0], y_offset + img_size[1]))
        else:
            bg = bg.resize(img_size)

        labels = []
        num_plants = random.randint(1, max_plants_per_image)

        # 2. Add foregrounds
        for _ in range(num_plants):
            fg_file = random.choice(fg_files)
            fg = Image.open(fg_file).convert("RGBA")

            # Augment foreground: Rotation
            angle = random.uniform(0, 360)
            fg = fg.rotate(angle, expand=True)

            # Augment foreground: Scaling (plant should take up 15-40% of the image width)
            scale = random.uniform(0.15, 0.40)
            new_w = int(img_size[0] * scale)
            # preserve aspect ratio
            new_h = int(fg.height * (new_w / fg.width))
            fg = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Augment foreground: Brightness (to simulate shadows/sunlight)
            enhancer = ImageEnhance.Brightness(fg)
            fg = enhancer.enhance(random.uniform(0.7, 1.2))

            # Position
            paste_x = random.randint(0, img_size[0] - new_w)
            paste_y = random.randint(0, img_size[1] - new_h)

            # Paste
            bg.alpha_composite(fg, dest=(paste_x, paste_y))

            # 3. Calculate YOLO Bounding Box
            # YOLO format: class_id center_x center_y width height (normalized 0-1)
            # Note: We use the full pasted region. Since we rotated with expand=True,
            # this is a slightly generous bounding box, which is good for detection.
            center_x = (paste_x + new_w / 2.0) / img_size[0]
            center_y = (paste_y + new_h / 2.0) / img_size[1]
            norm_w = new_w / img_size[0]
            norm_h = new_h / img_size[1]

            class_id = 0 # 0 for ragwort
            labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        # 4. Save Image and Label
        # Convert back to RGB for saving as JPG
        final_img = bg.convert("RGB")
        img_filename = f"synth_{i:04d}.jpg"
        txt_filename = f"synth_{i:04d}.txt"

        final_img.save(out_images / img_filename, quality=90)
        
        with open(out_labels / txt_filename, "w") as f:
            f.write("\n".join(labels))

    # Generate dataset.yaml for YOLO
    yaml_content = f"""path: {Path(output_dir).absolute().as_posix()}
train: images/train
val: images/train  # For now, using train as val just to check if it learns

names:
  0: ragwort
"""
    with open(Path(output_dir) / "dataset.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"\nDone! Dataset generated at: {output_dir}")
    print(f"YOLO config file ready at: {Path(output_dir) / 'dataset.yaml'}")

if __name__ == "__main__":
    # Create required folders if they don't exist
    Path("dataset_builder/backgrounds").mkdir(parents=True, exist_ok=True)
    Path("dataset_builder/foregrounds").mkdir(parents=True, exist_ok=True)
    
    # create_synthetic_dataset()
    print("Folders 'dataset_builder/backgrounds' and 'dataset_builder/foregrounds' have been created.")
    print("Place your images there, uncomment create_synthetic_dataset() in the code, and run this script.")
