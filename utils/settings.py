from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    forge_base_url: str
    model_checkpoint: str
    use_controlnet: bool
    controlnet_module: str
    controlnet_model: str
    temp_dir: Path
    output_dir: Path
    max_workers: int

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            forge_base_url=os.getenv("FORGE_BASE_URL", "http://127.0.0.1:7860").rstrip("/"),
            model_checkpoint=os.getenv("MODEL_PATH", "/models/sdxl-inpaint.ckpt"),
            use_controlnet=os.getenv("USE_CONTROLNET", "0") == "1",
            controlnet_module=os.getenv("CONTROLNET_MODULE", "reference_only"),
            controlnet_model=os.getenv("CONTROLNET_MODEL", "controlnetReference"),
            temp_dir=Path(os.getenv("TEMP_DIR", str(Path(__file__).resolve().parents[2] / "temp"))),
            output_dir=Path(os.getenv("OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs"))),
            max_workers=int(os.getenv("MAX_WORKERS", "1")),
        )


