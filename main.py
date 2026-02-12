import base64
import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from starlette.datastructures import UploadFile

from utils.jobs import JobManager
from utils.settings import Settings


settings = Settings.from_env()
app = FastAPI(title="Virtual Try-On API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
)

# Ensure directories exist
settings.temp_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)

app.mount("/results", StaticFiles(directory=str(settings.output_dir)), name="results")

jobs = JobManager(settings=settings)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(request: Request):
    """
    Upload endpoint that does NOT require python-multipart.

    Accepts JSON:
      {
        "person_image_b64": "data:image/png;base64,... or raw base64",
        "clothing_image_b64": "data:image/png;base64,... or raw base64"
      }

    Returns temp file paths.
    """
    ct = (request.headers.get("content-type") or "").lower()
    person_path = settings.temp_dir / f"person_{jobs.new_id()}.png"
    clothing_path = settings.temp_dir / f"clothing_{jobs.new_id()}.png"

    if "multipart/form-data" in ct:
        try:
            form = await request.form()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail='Multipart parsing unavailable (missing "python-multipart"). Use JSON base64 instead.',
            )

        person_up = form.get("person_image")
        clothing_up = form.get("clothing_image")
        if not isinstance(person_up, UploadFile) or not isinstance(clothing_up, UploadFile):
            raise HTTPException(status_code=400, detail="Missing person_image or clothing_image")

        await _save_uploadfile_png(person_up, person_path, force_mode="RGB")
        await _save_uploadfile_png(clothing_up, clothing_path, force_mode="RGBA")
        return {"person_path": str(person_path), "clothing_path": str(clothing_path)}

    if "application/json" in ct:
        body = await request.json()
        person_b64 = body.get("person_image_b64")
        clothing_b64 = body.get("clothing_image_b64")
        if not person_b64 or not clothing_b64:
            raise HTTPException(status_code=400, detail="Missing person_image_b64 or clothing_image_b64")

        person_img = _decode_b64_image(person_b64).convert("RGB")
        clothing_img = _decode_b64_image(clothing_b64).convert("RGBA")
        person_img.save(person_path, format="PNG", optimize=True)
        clothing_img.save(clothing_path, format="PNG", optimize=True)
        return {"person_path": str(person_path), "clothing_path": str(clothing_path)}

    raise HTTPException(
        status_code=415,
        detail="Unsupported content-type. Use multipart/form-data or application/json.",
    )


@app.post("/generate")
async def generate(request: Request):
    """
    Start a background try-on generation job.

    Accepts JSON (recommended; no python-multipart dependency):
      {
        "denoise_level": 0.65,
        "person_image_b64": "...",
        "clothing_image_b64": "..."
      }
    OR:
      {
        "denoise_level": 0.65,
        "person_image_path": "F:/.../person.png",
        "clothing_image_path": "F:/.../clothing.png"
      }
    """
    ct = (request.headers.get("content-type") or "").lower()

    person_path: Path
    clothing_path: Path
    denoise_level: float

    if "multipart/form-data" in ct:
        try:
            form = await request.form()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail='Multipart parsing unavailable (missing "python-multipart"). Use JSON base64 instead.',
            )

        denoise_level = float(form.get("denoise_level", 0.65))
        if denoise_level not in (0.50, 0.65, 0.75):
            raise HTTPException(status_code=400, detail="denoise_level must be one of 0.50, 0.65, 0.75")

        person_up = form.get("person_image")
        clothing_up = form.get("clothing_image")

        if isinstance(person_up, UploadFile) and isinstance(clothing_up, UploadFile):
            person_path = settings.temp_dir / f"person_{jobs.new_id()}.png"
            clothing_path = settings.temp_dir / f"clothing_{jobs.new_id()}.png"
            await _save_uploadfile_png(person_up, person_path, force_mode="RGB")
            await _save_uploadfile_png(clothing_up, clothing_path, force_mode="RGBA")
        else:
            person_image_path = form.get("person_image_path")
            clothing_image_path = form.get("clothing_image_path")
            if not (person_image_path and clothing_image_path):
                raise HTTPException(
                    status_code=400,
                    detail="Provide either person_image+clothing_image OR person_image_path+clothing_image_path.",
                )
            person_path = Path(str(person_image_path))
            clothing_path = Path(str(clothing_image_path))
            if not person_path.exists():
                raise HTTPException(status_code=400, detail="person_image_path does not exist")
            if not clothing_path.exists():
                raise HTTPException(status_code=400, detail="clothing_image_path does not exist")

        job_id = jobs.create_job(person_path=person_path, clothing_path=clothing_path, denoise=denoise_level)
        return {"job_id": job_id}

    if "application/json" in ct:
        body = await request.json()
        denoise_level = float(body.get("denoise_level", 0.65))
        if denoise_level not in (0.50, 0.65, 0.75):
            raise HTTPException(status_code=400, detail="denoise_level must be one of 0.50, 0.65, 0.75")

        if body.get("person_image_b64") and body.get("clothing_image_b64"):
            person_img = _decode_b64_image(body["person_image_b64"]).convert("RGB")
            clothing_img = _decode_b64_image(body["clothing_image_b64"]).convert("RGBA")
            person_path = settings.temp_dir / f"person_{jobs.new_id()}.png"
            clothing_path = settings.temp_dir / f"clothing_{jobs.new_id()}.png"
            person_img.save(person_path, format="PNG", optimize=True)
            clothing_img.save(clothing_path, format="PNG", optimize=True)
        else:
            person_image_path = body.get("person_image_path")
            clothing_image_path = body.get("clothing_image_path")
            if not (person_image_path and clothing_image_path):
                raise HTTPException(
                    status_code=400,
                    detail="Provide either person_image_b64+clothing_image_b64 OR person_image_path+clothing_image_path.",
                )
            person_path = Path(person_image_path)
            clothing_path = Path(clothing_image_path)
            if not person_path.exists():
                raise HTTPException(status_code=400, detail="person_image_path does not exist")
            if not clothing_path.exists():
                raise HTTPException(status_code=400, detail="clothing_image_path does not exist")

        job_id = jobs.create_job(person_path=person_path, clothing_path=clothing_path, denoise=denoise_level)
        return {"job_id": job_id}

    raise HTTPException(status_code=415, detail="Unsupported content-type. Use multipart/form-data or application/json.")



@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict(base_results_url="/results")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    # Keep response safe; detailed logs stay server-side.
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})


def _ext(filename: str) -> str:
    if not filename:
        return ".png"
    ext = os.path.splitext(filename)[1].lower()
    return ext if ext in (".jpg", ".jpeg", ".png") else ".png"

def _decode_b64_image(b64: str) -> Image.Image:
    """
    Accepts either:
      - raw base64
      - data URL: data:image/png;base64,<...>
    Returns a PIL image with EXIF orientation applied.
    """
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=False)
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")


async def _save_uploadfile_png(up: UploadFile, dest: Path, force_mode: str) -> None:
    data = await up.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert(force_mode)
        img.save(dest, format="PNG", optimize=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image upload")


