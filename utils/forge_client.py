from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class ForgeConfig:
    base_url: str
    model_checkpoint: str
    timeout_s: int = 300


class ForgeClient:
    """
    Minimal A1111/Forge REST client.

    Forge is expected to expose:
      - POST {base_url}/sdapi/v1/img2img
      - GET  {base_url}/sdapi/v1/options
      - POST {base_url}/sdapi/v1/options  (optional; for setting checkpoint)
    """

    def __init__(self, cfg: ForgeConfig):
        self.cfg = cfg
        self._session = requests.Session()
        # In some corporate / Windows setups, HTTP(S)_PROXY env vars can break localhost calls.
        # We explicitly ignore proxy env to make Forge/local calls reliable.
        self._session.trust_env = False

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.cfg.base_url}/sdapi/v1/options", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def set_checkpoint_best_effort(self) -> None:
        # A1111 uses 'sd_model_checkpoint' in options; Forge typically follows.
        try:
            self._session.post(
                f"{self.cfg.base_url}/sdapi/v1/options",
                json={"sd_model_checkpoint": self.cfg.model_checkpoint},
                timeout=10,
            )
        except Exception:
            # Not fatal: some setups disallow changing model via API.
            return

    def inpaint_img2img(
        self,
        *,
        init_image_b64_png: str,
        mask_b64_png: str,
        denoising_strength: float,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        width: int,
        height: int,
        seed: int = -1,
        controlnet_payload: Optional[Dict[str, Any]] = None,
        inpaint_full_res: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "init_images": [init_image_b64_png],
            "mask": mask_b64_png,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "denoising_strength": denoising_strength,
            "width": width,
            "height": height,
            "inpainting_mask_invert": 0,  # 0 = paint white area
            "inpainting_fill": 1 if denoising_strength <= 0.65 else 2,  # preserve vs stronger rewrite
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": 32,
            "mask_blur": 8,
        }

        if controlnet_payload:
            payload["alwayson_scripts"] = {"controlnet": controlnet_payload}

        url = f"{self.cfg.base_url}/sdapi/v1/img2img"
        started = time.time()
        r = self._session.post(url, json=payload, timeout=self.cfg.timeout_s)
        dt = time.time() - started
        if r.status_code != 200:
            raise RuntimeError(f"Forge img2img failed ({r.status_code}) after {dt:.1f}s: {r.text[:500]}")
        return r.json()


