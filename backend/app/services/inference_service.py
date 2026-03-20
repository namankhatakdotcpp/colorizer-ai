from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image
import torch

from app.config import settings
from inference_pipeline import ModularInferencePipeline

logger = logging.getLogger(__name__)


def _parse_stage_override(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    stages = [part.strip() for part in raw.split(",") if part.strip()]
    return stages or None


class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = bool(settings.USE_HALF_PRECISION)

        stage_override = _parse_stage_override(settings.PIPELINE_STAGES)
        if stage_override is None and not settings.ENABLE_FULL_PIPELINE:
            # Backward-compatible default API behavior: colorization only.
            stage_override = ["colorizer"]

        self.pipeline = ModularInferencePipeline(
            checkpoints_dir=Path("checkpoints"),
            config_path=Path(settings.PIPELINE_CONFIG_PATH),
            stage_override=stage_override,
            device=self.device,
            amp_enabled=self.amp_enabled,
        )

        logger.info(
            "InferenceService initialized | device=%s | stages=%s | config=%s",
            self.device,
            self.pipeline.stage_names,
            settings.PIPELINE_CONFIG_PATH,
        )

    async def colorize_async(self, file_bytes: bytes, request_id: str):
        del request_id  # request_id is kept for API logging consistency.

        img = Image.open(io.BytesIO(file_bytes))
        rgb = self.pipeline.process_image(img)

        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        result_img = Image.fromarray(rgb_u8)

        output_buffer = io.BytesIO()
        result_img.save(output_buffer, format="JPEG", quality=95)
        return output_buffer.getvalue()

    def graceful_shutdown(self):
        return None
