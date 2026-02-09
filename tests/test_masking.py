import numpy as np
from PIL import Image, ImageDraw

from utils.masking import generate_torso_mask


def test_generate_torso_mask_non_empty_on_synthetic_person():
    # Synthetic "person": head circle + body rectangle on plain bg.
    w, h = 512, 768
    img = Image.new("RGB", (w, h), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.ellipse((220, 60, 292, 132), fill=(200, 160, 140))  # head-ish
    d.rectangle((200, 130, 312, 420), fill=(60, 80, 120))  # torso-ish

    res = generate_torso_mask(img)
    m = np.array(res.mask.convert("L"))

    # Mask should cover a reasonable area.
    assert m.mean() > 5
    assert m.max() == 255 or m.max() > 200


def test_generate_torso_mask_dimensions_match_input():
    img = Image.new("RGB", (320, 240), (128, 128, 128))
    res = generate_torso_mask(img)
    assert res.mask.size == img.size


