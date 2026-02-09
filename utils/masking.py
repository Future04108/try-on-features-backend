from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskResult:
    mask: Image.Image  # "L" mode, white=edit area
    debug: dict


def generate_torso_mask(person_rgb: Image.Image) -> MaskResult:
    """
    Create a soft torso mask suitable for SDXL inpainting.

    Strategy (best-effort, no extra installs required):
    - Try MediaPipe pose if available (best geometry).
    - Else use face detection anchor to estimate torso ROI.
    - Optionally constrain mask with rembg matte if available.
    - Feather edges to avoid seams.
    """
    debug: dict = {}

    mp_mask = _try_mediapipe_pose_torso(person_rgb)
    if mp_mask is not None:
        debug["method"] = "mediapipe_pose"
        mask = mp_mask
    else:
        debug["method"] = "opencv_face_anchor"
        mask = _face_anchored_torso_mask(person_rgb, debug=debug)

    # Optional: constrain to person silhouette if rembg is available.
    matte = _try_rembg_matte(person_rgb)
    if matte is not None:
        debug["rembg"] = True
        mask = _intersect_with_matte(mask, matte)
    else:
        debug["rembg"] = False

    mask = _soften_mask(mask)

    # Safety: ensure non-empty mask.
    m = np.array(mask, dtype=np.uint8)
    if int(m.mean()) < 3:  # basically empty
        raise RuntimeError("Masking failed: torso mask is empty")

    return MaskResult(mask=mask, debug=debug)


def _try_mediapipe_pose_torso(person_rgb: Image.Image) -> Optional[Image.Image]:
    try:
        import mediapipe as mp  # type: ignore
        import cv2  # type: ignore
    except Exception:
        return None

    img = np.array(person_rgb.convert("RGB"))
    h, w = img.shape[:2]

    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    pose.close()
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    # Landmarks: shoulders + hips define a quadrilateral.
    # indices: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    # We use: left/right shoulder (11/12), left/right hip (23/24)
    idx = {"ls": 11, "rs": 12, "lh": 23, "rh": 24}
    pts = []
    for k in ("ls", "rs", "rh", "lh"):
        p = lm[idx[k]]
        if p.visibility < 0.3:
            return None
        pts.append((int(p.x * w), int(p.y * h)))

    # Expand slightly downward and sideways for clothing coverage.
    pts = _expand_polygon(pts, w=w, h=h, expand_px=int(0.04 * max(w, h)))

    mask = Image.new("L", (w, h), 0)
    try:
        from PIL import ImageDraw

        d = ImageDraw.Draw(mask)
        d.polygon(pts, fill=255)
    except Exception:
        return None

    return mask


def _expand_polygon(pts, w: int, h: int, expand_px: int):
    # Simple expansion: move points away from centroid.
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    out = []
    for x, y in pts:
        vx = x - cx
        vy = y - cy
        n = max(1e-6, (vx * vx + vy * vy) ** 0.5)
        ex = x + int(round(expand_px * vx / n))
        ey = y + int(round(expand_px * vy / n)) + int(round(0.5 * expand_px))
        out.append((int(np.clip(ex, 0, w - 1)), int(np.clip(ey, 0, h - 1))))
    return out


def _face_anchored_torso_mask(person_rgb: Image.Image, debug: dict) -> Image.Image:
    import cv2  # type: ignore

    img = np.array(person_rgb.convert("RGB"))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        # Fallback: center torso zone.
        debug["face"] = None
        return _center_fallback_mask(w=w, h=h)

    # Choose largest face.
    fx, fy, fw, fh = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    debug["face"] = {"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)}

    # Torso ROI: below face, width about 2.8x face, height about 3.3x face.
    cx = fx + fw / 2
    top = fy + int(0.9 * fh)
    torso_w = int(2.8 * fw)
    torso_h = int(3.3 * fh)

    left = int(max(0, cx - torso_w / 2))
    right = int(min(w, cx + torso_w / 2))
    bottom = int(min(h, top + torso_h))

    # If it looks like a full-body image, expand down a bit more.
    if bottom < int(0.75 * h):
        bottom = int(min(h, bottom + 0.10 * h))

    mask = np.zeros((h, w), dtype=np.uint8)

    # Blend of rectangle + ellipse for shoulders/chest.
    cv2.rectangle(mask, (left, top), (right, bottom), 255, thickness=-1)
    ell_center = (int(cx), int(top + 0.22 * (bottom - top)))
    ell_axes = (int(0.60 * (right - left) / 2), int(0.33 * (bottom - top)))
    cv2.ellipse(mask, ell_center, ell_axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    # Remove face/neck region to avoid painting over head.
    cv2.rectangle(mask, (int(fx - 0.15 * fw), int(fy - 0.10 * fh)), (int(fx + 1.15 * fw), int(fy + 1.30 * fh)), 0, -1)

    return Image.fromarray(mask, mode="L")


def _center_fallback_mask(w: int, h: int) -> Image.Image:
    import cv2  # type: ignore

    mask = np.zeros((h, w), dtype=np.uint8)
    # Conservative center torso region.
    left = int(0.20 * w)
    right = int(0.80 * w)
    top = int(0.18 * h)
    bottom = int(0.70 * h)
    cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    cv2.ellipse(mask, (w // 2, int(0.28 * h)), (int(0.30 * w), int(0.20 * h)), 0, 0, 360, 255, -1)
    return Image.fromarray(mask, mode="L")


def _try_rembg_matte(person_rgb: Image.Image) -> Optional[Image.Image]:
    try:
        from rembg import remove  # type: ignore
    except Exception:
        return None

    # rembg returns an RGBA with background removed.
    try:
        rgba = remove(person_rgb.convert("RGBA"))
        if isinstance(rgba, Image.Image):
            img = rgba
        else:
            # Some rembg variants return bytes
            img = Image.open(rgba).convert("RGBA")
        alpha = img.split()[-1].convert("L")
        return alpha
    except Exception:
        return None


def _intersect_with_matte(mask_l: Image.Image, matte_l: Image.Image) -> Image.Image:
    m = np.array(mask_l.convert("L"), dtype=np.uint8)
    a = np.array(matte_l.convert("L"), dtype=np.uint8)
    out = (m.astype(np.uint16) * (a > 10).astype(np.uint16)).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def _soften_mask(mask_l: Image.Image) -> Image.Image:
    import cv2  # type: ignore

    m = np.array(mask_l.convert("L"), dtype=np.uint8)
    # Dilation then erosion to close holes and stabilize borders.
    k = max(3, (min(m.shape[0], m.shape[1]) // 180) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.dilate(m, kernel, iterations=2)
    m = cv2.erode(m, kernel, iterations=1)

    # Feather edges for natural blending.
    blur = max(7, (min(m.shape[0], m.shape[1]) // 120) | 1)
    m = cv2.GaussianBlur(m, (blur, blur), sigmaX=0)

    # Keep within [0,255]
    m = np.clip(m, 0, 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")


def save_mask(mask: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.convert("L").save(path, format="PNG", optimize=True)


def compute_clothing_bbox(clothing_rgba: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    If clothing has alpha, find tight bbox of non-transparent pixels.
    Returns (left, top, right, bottom) in pixel coords.
    """
    if clothing_rgba.mode != "RGBA":
        clothing_rgba = clothing_rgba.convert("RGBA")
    alpha = np.array(clothing_rgba.split()[-1], dtype=np.uint8)
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return None
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    return left, top, right, bottom


