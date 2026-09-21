import os
import time
import requests
from pathlib import Path
from duckduckgo_search import DDGS
from rembg import remove
from PIL import Image
import io

def download_images(query, num_images, save_dir, prefix="img"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    downloaded = []
    
    print(f"Searching for: {query}")
    with DDGS() as ddgs:
        results = list(ddgs.images(
            keywords=query,
            region="wt-wt",
            safesearch="off",
            size="Medium",
            color="color",
            type_image="photo",
            max_results=num_images * 2  # Request more in case some fail
        ))
    
    count = 0
    for res in results:
        if count >= num_images:
            break
        try:
            url = res['image']
            img_data = requests.get(url, timeout=5).content
            
            # Verify it's a valid image
            img = Image.open(io.BytesIO(img_data))
            
            filename = Path(save_dir) / f"{prefix}_{count}.jpg"
            # save as RGB to handle weird formats
            img.convert("RGB").save(filename, "JPEG")
            downloaded.append(filename)
            count += 1
            print(f"Downloaded {count}/{num_images}")
        except Exception as e:
            print(f"Failed to download image: {e}")
            
    return downloaded

def process_foregrounds(input_files, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(input_files):
        try:
            print(f"Removing background for {file_path.name}...")
            input_img = Image.open(file_path)
            
            # Apply rembg
            output_img = remove(input_img)
            
            # Save as PNG
            out_name = Path(output_dir) / f"fg_{i}.png"
            output_img.save(out_name, "PNG")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

def process_backgrounds(input_files, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(input_files):
        try:
            img = Image.open(file_path)
            out_name = Path(output_dir) / f"bg_{i}.jpg"
            img.convert("RGB").save(out_name, "JPEG")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    import shutil
    
    # 1. Setup temporary dirs
    temp_fg_dir = "temp_dl_fg"
    temp_bg_dir = "temp_dl_bg"
    
    final_fg_dir = "dataset_builder/foregrounds"
    final_bg_dir = "dataset_builder/backgrounds"
    
    print("=== Downloading Foreground Images (Ragwort Rosettes & Flowers) ===")
    fg_files_1 = download_images("ragwort rosette top down isolated", 6, temp_fg_dir, "rosette")
    fg_files_2 = download_images("Senecio jacobaea flower isolated white background", 6, temp_fg_dir, "flower")
    
    all_fgs = fg_files_1 + fg_files_2
    
    print("\n=== Downloading Background Images (Grass) ===")
    bg_files_1 = download_images("grass lawn top down texture seamless", 5, temp_bg_dir, "grass1")
    bg_files_2 = download_images("wild meadow grass top down texture", 5, temp_bg_dir, "grass2")
    
    all_bgs = bg_files_1 + bg_files_2
    
    print("\n=== Removing Backgrounds with AI (rembg) ===")
    process_foregrounds(all_fgs, final_fg_dir)
    
    print("\n=== Copying Backgrounds ===")
    process_backgrounds(all_bgs, final_bg_dir)
    
    print("\n=== Cleanup ===")
    try:
        shutil.rmtree(temp_fg_dir)
        shutil.rmtree(temp_bg_dir)
    except:
        pass
        
    print("\nAll done! You can now run python src/scripts/generate_synthetic_dataset.py")
