import os
import urllib.request
import zipfile
import random
import shutil

def download_and_extract_coco(data_dir="data", num_images=15000):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "train2017.zip")
    extract_dir = os.path.join(data_dir, "train2017")
    final_dir = os.path.join(data_dir, "training_images")
    
    if os.path.exists(final_dir) and len(os.listdir(final_dir)) >= num_images:
        print(f"Dataset already acquired. Found {len(os.listdir(final_dir))} images in {final_dir}.")
        return

    if not os.path.exists(zip_path):
        url = "http://images.cocodataset.org/zips/train2017.zip"
        print(f"Downloading COCO train2017 via aria2c from {url} (Multi-threaded chunks...)")
        
        import subprocess
        
        # We must delete any corrupted partial downloads
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        subprocess.run(["aria2c", "-x", "16", "-s", "16", "-d", data_dir, "-o", "train2017.zip", url], check=True)
        print("Download complete.")
        
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")

    # Get all extracted image paths
    all_images = [f for f in os.listdir(extract_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total extracted images: {len(all_images)}")
    
    # Randomly select a subset
    random.seed(42)
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Move them to final destination
    os.makedirs(final_dir, exist_ok=True)
    print(f"Moving {len(selected_images)} selected images to {final_dir}...")
    for img in selected_images:
        src = os.path.join(extract_dir, img)
        dst = os.path.join(final_dir, img)
        shutil.move(src, dst)
        
    # Cleanup leftover zip and unselected extracted folder
    print("Cleaning up leftover files...")
    shutil.rmtree(extract_dir)
    os.remove(zip_path)
    
    final_count = len(os.listdir(final_dir))
    print(f"Dataset preparation complete! Extracted {final_count} images successfully to {final_dir}.")

if __name__ == "__main__":
    download_and_extract_coco(data_dir="data", num_images=15000)
