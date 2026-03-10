import io
import torch
import numpy as np
import logging
from PIL import Image
import warnings

# Avoid annoying skimage warnings on LAB conversion
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from skimage.color import rgb2lab, lab2rgb

from models.unet_colorizer import UNetColorizer

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self):
        self.device = torch.device("cpu")
        
        # Performance requirement: Keep inference on exactly 2 CPU threads
        torch.set_num_threads(2)
        
        self.model = UNetColorizer(in_channels=1, out_channels=2)
        
        # Load the checkpoint generated from Phase 2
        from app.config import settings
        model_path = settings.MODEL_WEIGHTS_PATH
        try:
            # We must load map_location CPU safely
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            logger.info("Successfully loaded UNet Colorizer weights!")
            # 🔎 VERIFY EXACT WEIGHTS FILE
            import hashlib

            logger.info(f"Loading weights from: {model_path}")

            with open(model_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            logger.info(f"Model MD5 hash: {file_hash}")
        except Exception as e:
            logger.warning(f"Could not load weights at {model_path}: {e}")

        self.model.to(self.device)
        self.model.eval()

        # Phase 4 constraints: Warmup pass
        self._warmup()

    def _warmup(self):
        logger.info("Running dummy warmup tensor through ResNet18...")
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 256, 256, device=self.device)
                self.model(dummy)
            logger.info("Warmup complete.")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    async def colorize_async(self, file_bytes: bytes, request_id: str):
        # 1. Image loading and RGB conversion
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        # 2. Resize to required 256x256 tensor size
        img_resized = img.resize((256, 256))
        img_np = np.array(img_resized) / 255.0

        # 3. LAB space mapping
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lab = rgb2lab(img_np)
            
        # Extract L channel and normalize to purely [0.0, 1.0] scale
        L = lab[:, :, 0] / 100.0

        # Format (B, C, H, W)
        L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # 4. Neural Network Inference
        with torch.no_grad():
            ab_pred = self.model(L_tensor) # Output is exactly in range [-1, 1] due to Tanh
        # DEBUG: Check predicted chroma range
        ab_debug = ab_pred.detach().cpu().numpy()
        print("DEBUG ab_pred min:", ab_debug.min())
        print("DEBUG ab_pred max:", ab_debug.max())
        print("DEBUG ab_pred mean:", ab_debug.mean())

        # Extract values
        ab_pred = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 5. Output Denormalization & Reconstruction
        lab_out = np.zeros((256, 256, 3))
        lab_out[:, :, 0] = L * 100         # Restore Lightness to [0, 100]
        # Mathematical constraint: multiply predicted ab (-1, 1) by 128 boundary
        lab_out[:, :, 1:] = ab_pred * 128 * 2.5

        # 6. Reconstruct to standard RGB 8-bit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_out = lab2rgb(lab_out)
            
        rgb_out = (rgb_out * 255.0).clip(0, 255).astype(np.uint8)

        # 7. Convert and return PNG image payload directly to frontend
        result_img = Image.fromarray(rgb_out)
        output_buffer = io.BytesIO()
        result_img.save(output_buffer, format="PNG")

        return output_buffer.getvalue()

    def graceful_shutdown(self):
        pass

   
