from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from utils.forge_client import ForgeClient, ForgeConfig
from utils.image_io import (
    clamp_size_keep_aspect,
    load_image_rgb,
    load_image_rgba,
    pil_to_base64_png,
    base64_png_to_pil,
)
from utils.masking import generate_torso_mask, save_mask
from utils.settings import Settings


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued|running|succeeded|failed
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    result_filename: Optional[str] = None
    debug: dict = field(default_factory=dict)

    def to_dict(self, base_results_url: str = "/results"):
        out = {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.result_filename:
            out["result_url"] = f"{base_results_url}/{self.result_filename}"
        if self.status in ("failed",):
            out["debug"] = self.debug
        return out


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)

        self._forge = ForgeClient(
            ForgeConfig(base_url=settings.forge_base_url, model_checkpoint=settings.model_checkpoint)
        )

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def create_job(self, *, person_path: Path, clothing_path: Path, denoise: float) -> str:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, status="queued", progress=0.0, message="Queued")
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run_job, job_id, person_path, clothing_path, denoise)
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, *, status: Optional[str] = None, progress: Optional[float] = None, message: str = ""):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status:
                job.status = status
            if progress is not None:
                job.progress = float(max(0.0, min(1.0, progress)))
            if message:
                job.message = message
            job.updated_at = time.time()

    def _run_job(self, job_id: str, person_path: Path, clothing_path: Path, denoise: float):
        try:
            self._update(job_id, status="running", progress=0.02, message="Loading images")

            person = load_image_rgb(person_path)
            clothing = load_image_rgba(clothing_path)

            self._update(job_id, progress=0.10, message="Generating torso mask")
            mask_res = generate_torso_mask(person)

            # Save debug mask to temp for troubleshooting (not exposed by default).
            mask_path = self.settings.temp_dir / f"mask_{job_id[:12]}.png"
            save_mask(mask_res.mask, mask_path)

            self._update(job_id, progress=0.18, message="Preparing inpaint inputs")

            # SDXL best practice: dimensions divisible by 8; keep within 1024.
            w, h = person.size
            nw, nh = clamp_size_keep_aspect(w, h, max_side=1024)
            nw, nh = (nw // 8) * 8, (nh // 8) * 8
            person_r = person.resize((nw, nh), Image.LANCZOS)
            mask_r = mask_res.mask.resize((nw, nh), Image.LANCZOS)

            init_b64 = pil_to_base64_png(person_r)
            mask_b64 = pil_to_base64_png(mask_r.convert("L"))

            # Build prompt: we can't guarantee IP-Adapter/ControlNet, but we can still
            # use a strong inpaint prompt tuned for clothing realism.
            prompt, negative = _build_prompts(denoise=denoise)

            controlnet = None
            if self.settings.use_controlnet:
                controlnet = _build_controlnet_reference_payload(
                    clothing_rgba=clothing,
                    module=self.settings.controlnet_module,
                    model=self.settings.controlnet_model,
                )

            self._update(job_id, progress=0.28, message="Calling SDXL Forge (inpaint)")
            self._forge.set_checkpoint_best_effort()

            try:
                resp = self._forge.inpaint_img2img(
                    init_image_b64_png=init_b64,
                    mask_b64_png=mask_b64,
                    denoising_strength=float(denoise),
                    prompt=prompt,
                    negative_prompt=negative,
                    steps=35,
                    cfg_scale=7.5,
                    sampler_name="Euler a",
                    width=nw,
                    height=nh,
                    controlnet_payload=controlnet,
                    inpaint_full_res=True,
                )
            except Exception as e:
                # If ControlNet payload breaks (missing extension/model), retry without it.
                if controlnet is not None:
                    self._update(job_id, progress=0.30, message="Retrying without ControlNet")
                    resp = self._forge.inpaint_img2img(
                        init_image_b64_png=init_b64,
                        mask_b64_png=mask_b64,
                        denoising_strength=float(denoise),
                        prompt=prompt,
                        negative_prompt=negative,
                        steps=35,
                        cfg_scale=7.5,
                        sampler_name="Euler a",
                        width=nw,
                        height=nh,
                        controlnet_payload=None,
                        inpaint_full_res=True,
                    )
                else:
                    raise e

            self._update(job_id, progress=0.88, message="Decoding result")

            images = resp.get("images") or []
            if not images:
                raise RuntimeError("Forge returned no images")

            out_img = base64_png_to_pil(images[0])
            out_filename = f"tryon_{job_id[:12]}.png"
            out_path = self.settings.output_dir / out_filename
            out_img.save(out_path, format="PNG", optimize=True)

            self._update(job_id, status="succeeded", progress=1.0, message="Done")
            with self._lock:
                self._jobs[job_id].result_filename = out_filename
                self._jobs[job_id].debug = {
                    "mask_path": str(mask_path),
                    "mask_debug": mask_res.debug,
                    "forge_base_url": self.settings.forge_base_url,
                }

        except Exception as e:
            self._update(job_id, status="failed", progress=1.0, message=str(e))
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.debug = {"error": str(e)}


def _build_prompts(*, denoise: float) -> tuple[str, str]:
    # Denoise controls how much garment/torso can change; with lower denoise we emphasize preserving pose.
    strength_hint = "preserve body pose and identity" if denoise <= 0.65 else "allow garment refit"

    prompt = (
        "photorealistic person, natural lighting, high detail fabric texture, "
        "realistic folds and seams, correct garment drape, "
        "shirt/top perfectly fitted to torso, "
        f"{strength_hint}, "
        "seamlessly blended edges, no artifacts, ultra sharp"
    )
    negative = (
        "cartoon, anime, illustration, CGI, plastic skin, blurry, lowres, "
        "deformed torso, extra arms, extra fingers, bad hands, "
        "text, watermark, logo, artifacts, halo, jagged edges, "
        "mismatched lighting, wrong perspective"
    )
    return prompt, negative


def _build_controlnet_reference_payload(*, clothing_rgba: Image.Image, module: str, model: str):
    """
    Best-effort ControlNet payload. Works only if Forge has ControlNet extension installed
    and the given module/model identifiers are valid for that environment.
    """
    # Many ControlNet APIs accept a list of args.
    # We send a minimal 'reference' conditioning image.
    clothing_b64 = pil_to_base64_png(clothing_rgba.convert("RGB"))
    return {
        "args": [
            {
                "enabled": True,
                "input_image": clothing_b64,
                "module": module,
                "model": model,
                "weight": 0.85,
                "resize_mode": 1,
                "guidance_start": 0.0,
                "guidance_end": 1.0,
                "pixel_perfect": True,
            }
        ]
    }


