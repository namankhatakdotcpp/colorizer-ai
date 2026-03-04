import os
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

# Assume architectures are fully trained and local
from models.unet_colorizer import UNetColorizer
from models.rrdb_sr import RRDBNet
from models.depth_model import DynamicFilterNetwork
from models.micro_contrast_model import MicroContrastModel

def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        print(f"Warning: Checkpoint '{path}' missing. Using untrained initialized weights.")
        return model
    
    checkpoint = torch.load(path, map_location=device)
    # The training architecture safely stripped the 'module.' wrapper, meaning these load natively!
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main(input_image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Inference Pipeline on {device}...")

    # 1. Boot up all 4 Neural Engines
    colorizer = load_checkpoint(UNetColorizer(1, 2).to(device), "checkpoints/stage1_colorizer_best.pth", device)
    sr_model = load_checkpoint(RRDBNet().to(device), "checkpoints/stage2_sr_best.pth", device)
    depth_model = load_checkpoint(DynamicFilterNetwork(3).to(device), "checkpoints/stage3_depth_best.pth", device)
    contrast_model = load_checkpoint(MicroContrastModel(3, 3).to(device), "checkpoints/stage4_contrast_best.pth", device)

    # 2. Read and format Input Image (Assume Grayscale/L-Channel start)
    raw_img = Image.open(input_image_path).convert('L') # Force grayscale
    l_tensor = ToTensor()(raw_img).unsqueeze(0).to(device) # Shape: (1, 1, H, W)

    with torch.no_grad():
        # --- STAGE 1: COLORIZATION ---
        print("Executing Stage 1: UNet Colorization...")
        ab_tensor = colorizer(l_tensor)
        # Synthetic LAB -> RGB differentiable merge (normally requires specific math bridging)
        rgb_tensor = torch.cat([l_tensor, ab_tensor], dim=1) # Shape: (1, 3, H, W)
        
        # --- STAGE 2: SUPER RESOLUTION ---
        print("Executing Stage 2: RRDB 4x Super Resolution...")
        hr_tensor = sr_model(rgb_tensor) # Shape: (1, 3, H*4, W*4)
        
        # --- STAGE 3: DEPTH MAPPING ---
        print("Executing Stage 3: DFN Depth Extraction...")
        depth_tensor = depth_model(hr_tensor)
        
        # --- STAGE 4: MICRO-CONTRAST ---
        print("Executing Stage 4: Laplacian Micro-Contrast Overdrive...")
        final_tensor = contrast_model(hr_tensor)
        
    # 3. Export Output
    final_image = ToPILImage()(final_tensor.squeeze(0).clamp(0, 1).cpu())
    out_path = os.path.join(output_dir, f"final_{os.path.basename(input_image_path)}")
    final_image.save(out_path)
    print(f"Pipeline Complete! Image saved to: {out_path}")

if __name__ == "__main__":
    # Example test trigger
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python inference_pipeline.py <path_to_grayscale_image.jpg>")
