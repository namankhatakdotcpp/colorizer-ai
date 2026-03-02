import io
import torch
import numpy as np
from PIL import Image
from skimage import color
from .colorizer import BasicColorizationModel
import torch.nn.functional as F

class ColorizerInference:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BasicColorizationModel().to(self.device)
        self.model.eval()
        
        # For production, we would load pre-trained weights here
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Loaded pre-trained weights from", model_path)
            except Exception as e:
                print("Failed to load weights:", e)
        else:
            print("Running in untrained mode (random weights). Output will show shapes/colors but will not be realistic until training is done.")

    def preprocess(self, img_pil: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Convert RGB image to LAB color space and extract L channel as tensor.
        """
        # Ensure image is RGB before conversion
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
            
        # Store original size to resize back later
        original_size = img_pil.size  # (W, H)
        
        # Resize to power-of-2 dimensions for our basic CNN (e.g., 256x256)
        img_resized = img_pil.resize((256, 256), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized)
        
        # Scale to [0, 1] and convert RGB to LAB
        img_np = img_np.astype(np.float32) / 255.0
        img_lab = color.rgb2lab(img_np)
        
        # Extract L channel. Range in skimage is [0, 100]. Let's normalize to [-1, 1] or [0, 1]
        # For standard CNN: normalize L to [-1, 1] roughly or center it.
        # Usually: L = L/50 - 1
        l_channel = img_lab[:, :, 0:1]
        l_norm = (l_channel / 50.0) - 1.0
        
        # Convert to tensor: (H, W, 1) -> (1, 1, H, W)
        l_tensor = torch.from_numpy(l_norm).permute(2, 0, 1).unsqueeze(0).float()
        
        return l_tensor.to(self.device), original_size

    def postprocess(self, output_ab: torch.Tensor, original_l: torch.Tensor, original_size: tuple[int, int]) -> Image.Image:
        """
        Combine original L channel with predicted AB channels, convert LAB back to RGB.
        """
        # Move inputs to CPU and numpy, remove batch dimension
        output_ab = output_ab.squeeze(0).cpu()  # (2, H, W)
        original_l = original_l.squeeze(0).cpu() # (1, H, W)
        
        # Un-normalize L to [0, 100]
        l_unnorm = (original_l + 1.0) * 50.0
        
        # Combile L and AB to (3, H, W)
        img_lab = torch.cat([l_unnorm, output_ab], dim=0)
        
        # shape to (H, W, 3)
        img_lab_np = img_lab.permute(1, 2, 0).numpy()
        
        # Convert LAB to RGB. skimage rgb2lab outputs RGB in [0, 1] range.
        # It handles value clamping automatically.
        with np.errstate(invalid='ignore'):
            img_rgb = color.lab2rgb(img_lab_np)
            
        # Rescale to [0, 255] for standard image saving
        img_rgb = (img_rgb * 255.0).clip(0, 255).astype(np.uint8)
        img_res = Image.fromarray(img_rgb)
        
        # Resize back to original dimensions
        img_res = img_res.resize(original_size, Image.Resampling.LANCZOS)
        
        return img_res

    @torch.no_grad()
    def process_image(self, img_pil: Image.Image) -> Image.Image:
        """
        Full pipeline: input image -> pre-process (L channel) -> model (predict ab) -> post-process (LAB to RGB image)
        """
        l_tensor, orig_size = self.preprocess(img_pil)
        
        # Predict A and B channels
        ab_pred = self.model(l_tensor)
        
        rendered_image = self.postprocess(ab_pred, l_tensor, orig_size)
        return rendered_image
