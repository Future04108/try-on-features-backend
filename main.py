import base64
import io
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from starlette.datastructures import UploadFile
from starlette.requests import ClientDisconnect

from utils.jobs import JobManager
from utils.settings import Settings


settings = Settings.from_env()
app = FastAPI(title="Virtual Try-On API", version="1.0.0")

# Wide‑open CORS for development; Caddy still enforces its own auth in front.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],      # includes Authorization and large headers
)

# Ensure directories exist
settings.temp_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)

app.mount("/results", StaticFiles(directory=str(settings.output_dir)), name="results")

jobs = JobManager(settings=settings)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


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
            print("[/upload] reading multipart form…")
            form = await request.form()
            print("[/upload] form read complete")
        except ClientDisconnect:
            print("[/upload] ClientDisconnect while reading multipart body")
            return {
                "status": "aborted",
                "message": "Upload cancelled due to disconnection - try smaller images or a faster connection",
            }
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
        return {"status": "success", "person_path": str(person_path), "clothing_path": str(clothing_path)}

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
        return {"status": "success", "person_path": str(person_path), "clothing_path": str(clothing_path)}

    raise HTTPException(
        status_code=415,
        detail="Unsupported content-type. Use multipart/form-data or application/json.",
    )


@app.post("/generate")
async def generate(request: Request):
    """
    Start a background try-on generation job.

    Accepts:
      - multipart/form-data with files person_image, clothing_image, denoise_level
      - application/json with base64 or file paths (see README)

    Returns JSON:
      { "status": "success", "job_id": "...", "message": "queued" }
      or error structure on failure.
    """
    ct = (request.headers.get("content-type") or "").lower()

    print("---- /generate request ----")
    print("Client:", request.client)
    try:
        print("Headers:", dict(request.headers))
    except Exception:
        pass

    person_path: Path
    clothing_path: Path
    denoise_level: float

    try:
        # multipart/form-data (FormData uploads)
        if "multipart/form-data" in ct:
            print("[/generate] Processing as multipart/form-data")
            try:
                print("[/generate] reading multipart form…")
                form = await request.form()
                print("[/generate] form read complete")
            except ClientDisconnect:
                print("[/generate] ClientDisconnect while reading multipart body")
                return {
                    "status": "aborted",
                    "message": "Upload cancelled due to disconnection - try smaller images or a faster connection",
                }
            except Exception as e:
                msg = str(e)
                print("[/generate] multipart parsing error:", type(e).__name__, msg)
                raise HTTPException(
                    status_code=400,
                    detail=f'Multipart parsing failed: {msg}. Ensure python-multipart is installed.',
                )

            try:
                denoise_level = float(form.get("denoise_level", 0.65))
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="denoise_level must be a valid number")
            if denoise_level not in (0.50, 0.65, 0.75):
                raise HTTPException(status_code=400, detail="denoise_level must be one of 0.50, 0.65, 0.75")

            person_up = form.get("person_image")
            clothing_up = form.get("clothing_image")
            if not person_up or not clothing_up:
                raise HTTPException(status_code=400, detail="Missing person_image or clothing_image in form data")

            # Read & log file sizes
            async def save_with_size(label: str, up: UploadFile, dest: Path, force_mode: str):
                print(f"[/generate] reading file {label}: {up.filename!r}")
                data = await up.read()
                size = len(data or b"")
                print(f"[/generate] read {size} bytes for {label}")
                if not data:
                    raise HTTPException(status_code=400, detail=f"Empty upload for {label}")
                try:
                    img = Image.open(io.BytesIO(data))
                    img = ImageOps.exif_transpose(img)
                    img = img.convert(force_mode)
                    img.save(dest, format="PNG", optimize=True)
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid image upload for {label}")

            if isinstance(person_up, UploadFile) and isinstance(clothing_up, UploadFile):
                person_path = settings.temp_dir / f"person_{jobs.new_id()}.png"
                clothing_path = settings.temp_dir / f"clothing_{jobs.new_id()}.png"
                await save_with_size("person_image", person_up, person_path, "RGB")
                await save_with_size("clothing_image", clothing_up, clothing_path, "RGBA")
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

        # application/json (base64 or paths)
        elif "application/json" in ct:
            print("[/generate] Processing as application/json")
            try:
                print("[/generate] reading JSON body…")
                raw = await request.body()
                print("[/generate] JSON body size:", len(raw or b""))
                import json

                body = json.loads(raw.decode("utf-8"))
            except ClientDisconnect:
                print("[/generate] ClientDisconnect while reading JSON body")
                return {
                    "status": "aborted",
                    "message": "Upload cancelled due to disconnection - try smaller images or a faster connection",
                }
            except Exception as e:
                msg = str(e)
                print("[/generate] error reading JSON body:", type(e).__name__, msg)
                raise HTTPException(status_code=400, detail=f"Error reading request body: {msg}")

            try:
                denoise_level = float(body.get("denoise_level", 0.65))
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="denoise_level must be a valid number")
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

        else:
            print("[/generate] Unsupported content-type:", ct)
            raise HTTPException(status_code=415, detail="Unsupported content-type. Use multipart/form-data or application/json.")

        # Queue background generation job
        started = datetime.utcnow()
        print(f"[/generate] Queuing Forge job at {started.isoformat()} - denoise={denoise_level}")
        try:
            job_id = jobs.create_job(
                person_path=person_path,
                clothing_path=clothing_path,
                denoise=denoise_level,
            )
        except Exception as e:
            print("[/generate] Error queuing job:", repr(e))
            # This is where Forge/SDXL errors would surface if they were synchronous
            return JSONResponse(
                status_code=500,
                content={
                    "status": "failed",
                    "error": str(e),
                    "message": "Generation failed - check server logs / Forge backend",
                },
            )

        ended = datetime.utcnow()
        print(f"[/generate] Job {job_id} queued at {ended.isoformat()} (queued in {(ended - started).total_seconds():.2f}s)")

        return {"status": "success", "job_id": job_id, "message": "queued"}
    except HTTPException:
        raise
    except ClientDisconnect:
        print("[/generate] ClientDisconnect outside body read")
        return {
            "status": "aborted",
            "message": "Client disconnected during request - try smaller images or a faster connection",
        }
    except Exception as e:
        print("[/generate] Unexpected error:", repr(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "message": "Generation failed - check server logs",
            },
        )


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        # Explicit "pending" vs 404 for better polling UX
        return {"status": "pending", "message": "Job not found yet"}

    data = job.to_dict(base_results_url="/results")

    # Ensure we always return a reasonable status/message
    status = data.get("status") or "pending"
    result_url = data.get("result_url")

    if status == "succeeded" and not result_url:
        data["status"] = "failed"
        data["message"] = data.get("message") or "No image generated - check input images or Forge backend"
    elif status not in ("succeeded", "failed", "pending"):
        data["status"] = status

    return data


@app.exception_handler(ClientDisconnect)
async def client_disconnect_handler(_, exc: ClientDisconnect):
    print("[global] ClientDisconnect caught:", exc)
    # Return 200 with an "aborted" status so the frontend doesn't see ERR_EMPTY_RESPONSE
    return JSONResponse(
        status_code=200,
        content={"status": "aborted", "message": "Client disconnected"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    # Keep response safe; detailed logs stay server-side.
    print("[global] Unhandled exception:", repr(exc))
    return JSONResponse(
        status_code=500,
        content={"status": "failed", "detail": f"Internal error: {type(exc).__name__}"},
    )


def _decode_b64_image(b64: str) -> Image.Image:
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