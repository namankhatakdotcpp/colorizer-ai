import torch
import cv2
import numpy as np
from models.unet_colorizer import UNetColorizer
from skimage.color import lab2rgb

# Load model
model = UNetColorizer(in_channels=1, out_channels=2)
ckpt = torch.load("checkpoints/stage1_colorizer_latest.pth", map_location="cpu")

# handle both formats
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.eval()

# Load image
img = cv2.imread("test.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize (VERY IMPORTANT)
img_gray = cv2.resize(img_gray, (256, 256))

# Normalize
l = img_gray.astype(np.float32) / 255.0

# Convert to tensor
l_tensor = torch.tensor(l).unsqueeze(0).unsqueeze(0)

# Forward pass
with torch.no_grad():
    ab = model(l_tensor).squeeze().permute(1, 2, 0).numpy()

# LAB reconstruction
lab = np.zeros((256, 256, 3))
lab[:, :, 0] = l * 100
lab[:, :, 1:] = ab * 128

# Convert to RGB
rgb = lab2rgb(lab)

# Save
rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
cv2.imwrite("output_stage1.jpg", rgb)

print("✅ Saved output_stage1.jpg")