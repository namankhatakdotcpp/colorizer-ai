import io
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from PIL import Image

from models.depth_model import DynamicFilterNetwork
from models.micro_contrast_model import MicroContrastModel
from models.rrdb_sr import RRDBNet
from models.unet_colorizer import UNetColorizer

# Avoid noisy skimage warnings during LAB conversion.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from skimage.color import lab2rgb, rgb2lab

logger = logging.getLogger(__name__)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


class InferenceService:
    def __init__(self):
        self.device = torch.device("cpu")
        torch.set_num_threads(2)

        self.stage1 = UNetColorizer(in_channels=1, out_channels=2).to(self.device).eval()
        self.stage2 = RRDBNet().to(self.device).eval()
        self.stage3 = DynamicFilterNetwork(in_channels=3).to(self.device).eval()
        self.stage4 = MicroContrastModel(in_channels=3, out_channels=3).to(self.device).eval()

        from app.config import settings

        model_path = Path(settings.MODEL_WEIGHTS_PATH)
        checkpoints_dir = model_path.parent if model_path.parent != Path("") else Path("checkpoints")
        self.enable_full_pipeline = os.environ.get("ENABLE_FULL_PIPELINE", "0").strip().lower() in {"1", "true", "yes"}

        self.stage1_loaded = self._load_checkpoint(
            self.stage1,
            candidates=(
                checkpoints_dir / "stage1_colorizer_best.pth",
                model_path,
                checkpoints_dir / "stage1_colorizer_latest.pth",
            ),
            stage_name="stage1_colorizer",
            required=True,
        )
        self.stage2_loaded = self._load_checkpoint(
            self.stage2,
            candidates=(checkpoints_dir / "stage2_sr_best.pth", checkpoints_dir / "stage2_sr_latest.pth"),
            stage_name="stage2_sr",
            required=False,
        )
        self.stage3_loaded = self._load_checkpoint(
            self.stage3,
            candidates=(checkpoints_dir / "stage3_depth_best.pth", checkpoints_dir / "stage3_depth_latest.pth"),
            stage_name="stage3_depth",
            required=False,
        )
        self.stage4_loaded = self._load_checkpoint(
            self.stage4,
            candidates=(checkpoints_dir / "stage4_contrast_best.pth", checkpoints_dir / "stage4_contrast_latest.pth"),
            stage_name="stage4_contrast",
            required=False,
        )

        if self.enable_full_pipeline:
            missing = [
                name
                for name, loaded in (
                    ("stage2_sr", self.stage2_loaded),
                    ("stage3_depth", self.stage3_loaded),
                    ("stage4_contrast", self.stage4_loaded),
                )
                if not loaded
            ]
            if missing:
                raise RuntimeError(
                    "ENABLE_FULL_PIPELINE is set but required stage checkpoints are missing: "
                    + ", ".join(missing)
                )

        self._warmup()

    def _load_checkpoint(
        self,
        model: torch.nn.Module,
        candidates: Iterable[Path],
        stage_name: str,
        required: bool,
    ) -> bool:
        load_error: Optional[Exception] = None
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if not path.exists():
                continue

            try:
                checkpoint = torch.load(path, map_location=self.device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                state_dict = _strip_module_prefix(state_dict)
                model.load_state_dict(state_dict, strict=True)
                logger.info("Loaded %s checkpoint: %s", stage_name, path)
                return True
            except Exception as exc:
                load_error = exc
                logger.warning("Failed loading %s checkpoint at %s: %s", stage_name, path, exc)

        if required:
            raise RuntimeError(f"Missing valid checkpoint for {stage_name}. Last error: {load_error}")

        logger.warning("No checkpoint loaded for optional stage: %s", stage_name)
        return False

    def _warmup(self) -> None:
        logger.info("Running Stage1 warmup tensor...")
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 256, 256, device=self.device)
                _ = self.stage1(dummy)
            logger.info("Warmup complete.")
        except Exception as exc:
            logger.error("Warmup failed: %s", exc)

    def _run_optional_stages(self, rgb: np.ndarray) -> np.ndarray:
        if not self.enable_full_pipeline:
            return rgb

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.stage2(tensor) if self.stage2_loaded else tensor
            if self.stage3_loaded:
                _ = self.stage3(out)
            out = self.stage4(out) if self.stage4_loaded else out

        out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(out_np, 0.0, 1.0)

    async def colorize_async(self, file_bytes: bytes, request_id: str):
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_resized = img.resize((256, 256), Image.Resampling.BICUBIC)
        img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lab = rgb2lab(img_np)

        l_channel = lab[:, :, 0] / 100.0
        l_tensor = torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            ab_pred = self.stage1(l_tensor)
        ab_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

        lab_out = np.zeros((256, 256, 3), dtype=np.float32)
        lab_out[:, :, 0] = l_channel * 100.0
        lab_out[:, :, 1:] = np.clip(ab_np, -1.0, 1.0) * 128.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_out = lab2rgb(lab_out)

        rgb_out = np.clip(rgb_out, 0.0, 1.0)
        rgb_out = self._run_optional_stages(rgb_out)
        rgb_u8 = (rgb_out * 255.0).clip(0, 255).astype(np.uint8)

        result_img = Image.fromarray(rgb_u8)
        output_buffer = io.BytesIO()
        result_img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    def graceful_shutdown(self):
        return None

   
