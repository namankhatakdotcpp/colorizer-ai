import os
import glob
import numpy as np
from PIL import Image
from skimage import color
from pathlib import Path
from tqdm import tqdm

def preprocess_images_to_lab(input_dir, output_dir, img_size=(256, 256)):
    """
    Converts RGB images to LAB color space and saves L (lightness) and ab (color) 
    channels separately as fast-loading numpy arrays.
    
    Args:
        input_dir (str): Directory containing source RGB images.
        output_dir (str): Directory to save the preprocessed numpy arrays.
        img_size (tuple): Target resize dimensions (width, height) before conversion.
    """
    # Create output directories for L and ab channels
    l_dir = os.path.join(output_dir, 'L')
    ab_dir = os.path.join(output_dir, 'AB')
    os.makedirs(l_dir, exist_ok=True)
    os.makedirs(ab_dir, exist_ok=True)
    
    # Find all images in input directory
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Starting preprocessing...")
    
    for img_path in tqdm(image_paths, desc="Converting RGB to LAB"):
        try:
            # Load and resize RGB image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size, Image.BICUBIC)
            
            # Convert PIL image to numpy array (H, W, 3) in [0, 255]
            img_np = np.array(img)
            
            # Convert RGB to LAB using skimage
            # Note: skimage color.rgb2lab expects RGB in [0, 255] or [0, 1]
            # It returns L in [0, 100] and a,b roughly in [-128, 127]
            lab = color.rgb2lab(img_np)
            
            # Extract L and ab channels
            l_channel = lab[:, :, 0] # Shape: (H, W)
            ab_channel = lab[:, :, 1:] # Shape: (H, W, 2)
            
            # Get base filename without extension
            base_filename = Path(img_path).stem
            
            # Save as numpy arrays (.npy)
            # This is much faster to load during PyTorch training than reading JPEGs and converting on the fly
            np.save(os.path.join(l_dir, f"{base_filename}.npy"), l_channel.astype(np.float32))
            np.save(os.path.join(ab_dir, f"{base_filename}.npy"), ab_channel.astype(np.float32))
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            
    print(f"Preprocessing complete. Saved L and ab channels to {output_dir}")

if __name__ == "__main__":
    # Example usage:
    # Set your input directory containing original RGB training images
    input_images_dir = "data/colorization/rgb_images"
    
    # Set your output directory for the fast-loading numpy arrays
    output_numpy_dir = "dataset_lab"
    
    # Create dummy dir for example if it doesn't exist
    os.makedirs(input_images_dir, exist_ok=True)
    
    preprocess_images_to_lab(
        input_dir=input_images_dir,
        output_dir=output_numpy_dir,
        img_size=(256, 256)
    )
